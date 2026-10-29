import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import tqdm
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import clip

import libs.autoencoder as autoencoder
from libs.uvit import UViT
from semantic_likelihood import gaussian_entropy
from UViT.utils import (
    NoiseScheduleVP,
    stable_diffusion_beta_schedule,
    amortize,
    get_model_input_time,
    inverse_data_transform,
    seed_everything,
    conditioned_update,
    parse_args_and_config,
)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def main(args, config):

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.backends.cudnn.benchmark = True
    seed_everything(args.seed)

    image_size = config.dataset.image_size
    z_size = image_size // 8
    patch_size = 2 if image_size == 256 else 4
    total_n_samples = args.total_n_sample
    if total_n_samples % args.sample_batch_size != 0:
        raise ValueError(
            "Total samples for sampling must be divided exactly by args.sample_batch_size, but got {} and {}".format(
                total_n_samples, args.sample_batch_size
            )
        )
    if args.fixed_class == 10000:
        fixed_classes = torch.randint(low=0, high=1000, size=(total_n_samples,))
    else:
        fixed_classes = torch.full((total_n_samples,), args.fixed_class, dtype=torch.long)
    fixed_classes = fixed_classes.to(device)

    shard_size = (total_n_samples + args.num_shards - 1) // args.num_shards
    slot_start = args.shard_id * shard_size
    slot_end = min(slot_start + shard_size, total_n_samples)

    seed_everything(args.seed * 1_000_003 + args.shard_id + 1)

    ae = autoencoder.get_model(args.encoder_path)
    ae.to(device)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return ae.decode(_batch)

    # nnet = myUViT(
    nnet = UViT(
        img_size=z_size,
        patch_size=patch_size,
        in_chans=4,
        embed_dim=1152,
        depth=28,
        num_heads=16,
        num_classes=1001,
        conv=False,
    )

    nnet.to(device)
    nnet.load_state_dict(torch.load(args.uvit_path, map_location={"cuda:%d" % 0: "cuda:%d" % args.device}))
    nnet.eval()

    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    clip_mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    clip_std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    #########   get t sequence (note that t is different from timestep)  ##########
    betas = stable_diffusion_beta_schedule()
    ns = NoiseScheduleVP(schedule="discrete", betas=torch.tensor(betas, device=device).float())

    def get_time_steps(skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps, **recommended for DPM-Solver**.
                - 'time_uniform': uniform time for the time steps. (Used in DDIM and DDPM.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == "logSNR":
            lambda_T = ns.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = ns.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            # print(torch.min(torch.abs(logSNR_steps - self.noise_schedule.marginal_lambda(self.noise_schedule.inverse_lambda(logSNR_steps)))).item())
            return ns.inverse_lambda(logSNR_steps)
        elif skip_type == "t2":
            t_order = 2
            t = torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1).pow(t_order).to(device)
            return t
        elif skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == "time_quadratic":
            t = torch.linspace(t_0, t_T, 10000000).to(device)
            quadratic_t = torch.sqrt(t)
            quadratic_steps = torch.linspace(quadratic_t[0], quadratic_t[-1], N + 1).to(device)
            return torch.flip(
                torch.cat(
                    [t[torch.searchsorted(quadratic_t, quadratic_steps)[:-1]], t_T * torch.ones((1,)).to(device)],
                    dim=0,
                ),
                dims=[0],
            )
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type)
            )

    t_0 = 1.0 / ns.total_N
    t_T = ns.T

    base_half_steps = args.timesteps // 2
    screen_iters = args.mc_timesteps // 2

    t_seq_base = get_time_steps(skip_type=args.skip_type, t_T=t_T, t_0=t_0, N=base_half_steps, device=device)
    t_seq_base = torch.flip(t_seq_base, dims=[0])

    t_seq_screen = get_time_steps(skip_type=args.skip_type, t_T=t_T, t_0=t_0, N=screen_iters, device=device)
    t_seq_screen = torch.flip(t_seq_screen, dims=[0])

    def diffusion_sample(xT, classes, t_seq, num_half_steps):
        model_kwargs = {"y": classes}
        T = t_seq[num_half_steps]
        xt_next = xT
        eps_mu_t_next = nnet(xT, get_model_input_time(ns, T).expand(xT.shape[0]), **model_kwargs)
        for timestep in range(num_half_steps, 0, -1):
            s, t = t_seq[timestep], t_seq[timestep - 1]
            xt_next, _ = conditioned_update(
                ns=ns,
                x=xt_next,
                s=s,
                t=t,
                model=nnet,
                model_s=eps_mu_t_next,
                pre_wuq=False,
                r1=0.5,
                **model_kwargs,
            )
            eps_mu_t_next = nnet(xt_next, get_model_input_time(ns, t).expand(xt_next.shape[0]), **model_kwargs)
        return xt_next

    def decode_large_batch(_batch):
        if z_size == 32:
            decode_mini_batch_size = 8  # use a small batch size since the decoder is large
        else:
            decode_mini_batch_size = 1  # use a small batch size since the decoder is large
        xs = []
        pt = 0
        for _decode_mini_batch_size in amortize(_batch.size(0), decode_mini_batch_size):
            x = decode(_batch[pt : pt + _decode_mini_batch_size])
            pt += _decode_mini_batch_size
            xs.append(x)
        xs = torch.concat(xs, dim=0)
        assert xs.size(0) == _batch.size(0)
        return xs

    def clip_encode(images_01):
        imgs = F.interpolate(images_01, size=224, mode="bicubic", align_corners=False, antialias=True)
        imgs = imgs.clamp(0.0, 1.0)
        imgs = (imgs - clip_mean) / clip_std
        with torch.no_grad():
            feats = clip_model.encode_image(imgs)
        return feats.float()

    def screen_candidate(cand, classes_b):
        B = cand.shape[0]
        mc = args.mc_size
        noise = torch.randn(mc, B, 4, z_size, z_size, device=device)
        xT_pert = (cand.unsqueeze(0) + args.sigma_perturb * noise) / (1 + args.sigma_perturb ** 2) ** 0.5
        xT_pert_flat = xT_pert.reshape(mc * B, 4, z_size, z_size)
        classes_mc = classes_b.repeat(mc)

        screen_lat = diffusion_sample(xT_pert_flat, classes_mc, t_seq_screen, screen_iters)
        decoded_01 = inverse_data_transform(decode_large_batch(screen_lat))
        feats = clip_encode(decoded_01)
        feats = feats.reshape(mc, B, feats.shape[-1]).transpose(0, 1)
        return gaussian_entropy(feats.cpu().numpy(), sigma_squared=1e-3)

    def neighborhood_step(pivot, lam):
        eps = torch.randn_like(pivot)
        return math.sqrt(max(0.0, 1.0 - lam * lam)) * pivot + lam * eps

    #########   start sample  ##########
    exp_dir = f"{args.exp_path}/imagenet{image_size}/zo_fixed_class{args.fixed_class}_sigma{args.sigma_perturb}_S{args.mc_size}_K{args.zo_iters}_N{args.zo_neighborhood}_lam{args.zo_lambda}_S{args.seed}/"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(exp_dir + "0/imgs", exist_ok=True)

    shard_n = max(0, slot_end - slot_start)
    trajectory_log = np.full((shard_n, args.zo_iters + 1), np.nan, dtype=np.float32)

    pbar = tqdm.tqdm(total=shard_n, desc=f"shard {args.shard_id} filled slots")
    with torch.no_grad():
        for batch_start in range(slot_start, slot_end, args.sample_batch_size):
            batch_slots = list(range(batch_start, min(batch_start + args.sample_batch_size, slot_end)))
            B_eff = len(batch_slots)
            local_off = batch_start - slot_start
            classes = fixed_classes[torch.tensor(batch_slots, device=device)]

            pivot = torch.randn(B_eff, 4, z_size, z_size, device=device)
            pivot_score = screen_candidate(pivot, classes)
            trajectory_log[local_off : local_off + B_eff, 0] = pivot_score

            for k in range(args.zo_iters):
                cand_list = [pivot]
                score_list = [pivot_score]
                for _ in range(args.zo_neighborhood):
                    cand = neighborhood_step(pivot, args.zo_lambda)
                    cand_list.append(cand)
                    score_list.append(screen_candidate(cand, classes))

                scores_arr = np.stack(score_list, axis=0)
                cands_arr = torch.stack(cand_list, dim=0)
                best_idx = scores_arr.argmin(axis=0)
                slot_idx = torch.arange(B_eff, device=device)
                best_idx_t = torch.tensor(best_idx, device=device)
                pivot = cands_arr[best_idx_t, slot_idx]
                pivot_score = scores_arr[best_idx, np.arange(B_eff)]
                trajectory_log[local_off : local_off + B_eff, k + 1] = pivot_score

            xT = pivot
            model_kwargs = {"y": classes}
            timestep = base_half_steps

            ###### Initialize
            T = t_seq_base[timestep]
            xt_next = xT
            eps_mu_t_next = nnet(xT, get_model_input_time(ns, T).expand(xT.shape[0]), **model_kwargs)

            for timestep in range(base_half_steps, 0, -1):

                s, t = t_seq_base[timestep], t_seq_base[timestep - 1]

                xt_next, _ = conditioned_update(
                    ns=ns,
                    x=xt_next,
                    s=s,
                    t=t,
                    model=nnet,
                    model_s=eps_mu_t_next,
                    pre_wuq=False,
                    r1=0.5,
                    **model_kwargs,
                )
                eps_mu_t_next = nnet(xt_next, get_model_input_time(ns, t).expand(xt_next.shape[0]), **model_kwargs)

            x = inverse_data_transform(decode_large_batch(xt_next))

            # torch.save(x.cpu().numpy(), os.path.join(exp_dir + "0/", f"{batch_start}.pt"))

            x = x.cpu().numpy()
            for i in range(x.shape[0]):
                img = x[i].transpose(1, 2, 0)
                img = (img * 255).astype(np.uint8)
                img_pil = Image.fromarray(img)
                img_pil.save(os.path.join(exp_dir + "0/imgs", f"{batch_slots[i]:05d}.png"))

            pbar.update(B_eff)
            pbar.set_postfix(
                init_ent=f"{float(trajectory_log[local_off : local_off + B_eff, 0].mean()):.3f}",
                final_ent=f"{float(pivot_score.mean()):.3f}",
            )

    pbar.close()

    if args.num_shards == 1:
        traj_name = "zo_trajectory.npy"
    else:
        traj_name = f"zo_trajectory_shard{args.shard_id:04d}_of{args.num_shards:04d}_slots{slot_start:06d}-{slot_end:06d}.npy"
    np.save(os.path.join(exp_dir, traj_name), trajectory_log)

    return exp_dir


if __name__ == "__main__":
    args, config = parse_args_and_config()
    exp_dir = main(args, config)
    print(exp_dir, end="")
