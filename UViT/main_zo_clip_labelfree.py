import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import tqdm
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import open_clip

import libs.autoencoder as autoencoder
from libs.uvit import UViT
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
    print(args.seed)

    if args.num_shards < 1:
        raise ValueError(f"--num_shards must be >= 1, got {args.num_shards}.")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(f"--shard_id={args.shard_id} out of range for --num_shards={args.num_shards}.")
    if not (0.0 < args.zo_lambda <= 1.0):
        raise ValueError(f"--zo_lambda must be in (0, 1], got {args.zo_lambda}.")
    if args.zo_iters < 1 or args.zo_neighborhood < 1:
        raise ValueError("--zo_iters and --zo_neighborhood must be >= 1.")

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

    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    clip_model = clip_model.to(device).eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    W = open_clip.build_zero_shot_classifier(
        clip_model,
        clip_tokenizer,
        open_clip.IMAGENET_CLASSNAMES,
        open_clip.OPENAI_IMAGENET_TEMPLATES,
        device=device,
    )
    logit_scale = clip_model.logit_scale.exp().float()
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

    num_half_steps = args.timesteps // 2
    screen_iters = args.mc_timesteps // 2
    if screen_iters < 1:
        raise ValueError(f"After // 2, screen_iters={screen_iters} must be >= 1.")

    t_seq_base = get_time_steps(skip_type=args.skip_type, t_T=t_T, t_0=t_0, N=num_half_steps, device=device)
    t_seq_base = torch.flip(t_seq_base, dims=[0])

    t_seq_screen = get_time_steps(skip_type=args.skip_type, t_T=t_T, t_0=t_0, N=screen_iters, device=device)
    t_seq_screen = torch.flip(t_seq_screen, dims=[0])

    reuse_scored_latent = screen_iters == num_half_steps

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

    def clip_score_confidence(images_01):
        imgs = F.interpolate(images_01, size=224, mode="bicubic", align_corners=False, antialias=True)
        imgs = imgs.clamp(0.0, 1.0)
        imgs = (imgs - clip_mean) / clip_std
        feats = clip_model.encode_image(imgs).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = logit_scale * (feats @ W)
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        neg_entropy = (p * logp).sum(dim=-1)
        return neg_entropy

    def score_candidate(cand, classes_b):
        out_lat = diffusion_sample(cand, classes_b, t_seq_screen, screen_iters)
        decoded_01 = inverse_data_transform(decode_large_batch(out_lat))
        scores = clip_score_confidence(decoded_01)
        return scores.float().cpu().numpy(), out_lat

    def neighborhood_step(pivot, lam):
        eps = torch.randn_like(pivot)
        return math.sqrt(max(0.0, 1.0 - lam * lam)) * pivot + lam * eps

    #########   start sample  ##########
    exp_dir = f"{args.exp_path}/imagenet{image_size}/zo_verifier_clip_labelfree_fixed_class{args.fixed_class}_K{args.zo_iters}_N{args.zo_neighborhood}_lam{args.zo_lambda}_S{args.seed}/"
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
            pivot_score, pivot_lat = score_candidate(pivot, classes)
            trajectory_log[local_off : local_off + B_eff, 0] = pivot_score

            for k in range(args.zo_iters):
                cand_list = [pivot]
                score_list = [pivot_score]
                lat_list = [pivot_lat]
                for _ in range(args.zo_neighborhood):
                    cand = neighborhood_step(pivot, args.zo_lambda)
                    s, lat = score_candidate(cand, classes)
                    cand_list.append(cand)
                    score_list.append(s)
                    lat_list.append(lat)

                scores_arr = np.stack(score_list, axis=0)
                cands_arr = torch.stack(cand_list, dim=0)
                lats_arr = torch.stack(lat_list, dim=0)
                best_idx = scores_arr.argmax(axis=0)
                slot_idx = torch.arange(B_eff, device=device)
                best_idx_t = torch.tensor(best_idx, device=device)
                pivot = cands_arr[best_idx_t, slot_idx]
                pivot_lat = lats_arr[best_idx_t, slot_idx]
                pivot_score = scores_arr[best_idx, np.arange(B_eff)]
                trajectory_log[local_off : local_off + B_eff, k + 1] = pivot_score

            if reuse_scored_latent:
                final_lat = pivot_lat
            else:
                final_lat = diffusion_sample(pivot, classes, t_seq_base, num_half_steps)
            x = inverse_data_transform(decode_large_batch(final_lat))
            x = x.cpu().numpy()
            for i in range(x.shape[0]):
                img = x[i].transpose(1, 2, 0)
                img = (img * 255).astype(np.uint8)
                img_pil = Image.fromarray(img)
                img_pil.save(os.path.join(exp_dir + "0/imgs", f"{batch_slots[i]:05d}.png"))

            pbar.update(B_eff)
            pbar.set_postfix(
                init_negent=f"{float(trajectory_log[local_off : local_off + B_eff, 0].mean()):.4f}",
                final_negent=f"{float(pivot_score.mean()):.4f}",
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
