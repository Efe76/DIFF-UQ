import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tqdm
from PIL import Image
import numpy as np
import torch

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


def main(args, config):

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.backends.cudnn.benchmark = True
    print(args.seed)
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
    n_rounds = total_n_samples // args.sample_batch_size
    fixed_xT = torch.randn([args.total_n_sample, 4, z_size, z_size])
    if args.fixed_class == 10000:
        fixed_classes = torch.randint(low=0, high=1000, size=(args.sample_batch_size, n_rounds))
    else:
        fixed_classes = torch.randint(
            low=args.fixed_class, high=args.fixed_class + 1, size=(args.sample_batch_size, n_rounds)
        ).to(device)

    ae = autoencoder.get_model(args.encoder_path)
    ae.to(device)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return ae.decode(_batch)

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
    mc_half_steps = args.mc_timesteps // 2

    t_seq_base = get_time_steps(skip_type=args.skip_type, t_T=t_T, t_0=t_0, N=base_half_steps, device=device)
    t_seq_base = torch.flip(t_seq_base, dims=[0])
    t_seq_mc = get_time_steps(skip_type=args.skip_type, t_T=t_T, t_0=t_0, N=mc_half_steps, device=device)
    t_seq_mc = torch.flip(t_seq_mc, dims=[0])

    #########   start sample  ##########
    exp_dir = f"{args.exp_path}/imagenet{image_size}/input_perturb_fixed_class{args.fixed_class}_sigma{args.sigma_perturb}_S{args.mc_size}_S{args.seed}/"
    os.makedirs(exp_dir, exist_ok=True)

    M = args.mc_size + 1

    print(
        f"\n>>> Base image (folder 0): {args.timesteps} timesteps. "
        f"MC samples (folders 1..{args.mc_size}, sequential): {args.mc_timesteps} timesteps. "
        f"sigma={args.sigma_perturb}"
    )

    for m in range(M):
        folder = os.path.join(exp_dir, f"{m}/imgs")
        os.makedirs(folder, exist_ok=True)
        img_count = 0

        if m == 0:
            t_seq = t_seq_base
            num_half_steps = base_half_steps
        else:
            t_seq = t_seq_mc
            num_half_steps = mc_half_steps

        with torch.no_grad():
            for loop in tqdm.tqdm(range(n_rounds), desc=f"Folder {m}/{M-1}"):
                xT = fixed_xT[loop * args.sample_batch_size : (loop + 1) * args.sample_batch_size, :, :, :].to(device)

                if m > 0:
                    xT = (xT + args.sigma_perturb * torch.randn_like(xT)) / (1 + args.sigma_perturb ** 2) ** 0.5

                classes = fixed_classes[:, loop].to(device)
                model_kwargs = {"y": classes}
                timestep = num_half_steps

                ###### Initialize
                T = t_seq[timestep]
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

                x = inverse_data_transform(decode_large_batch(xt_next))

                x = x.cpu().numpy()
                for i in range(x.shape[0]):
                    img = x[i].transpose(1, 2, 0)
                    img = (img * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img)
                    img_pil.save(os.path.join(folder, f"{img_count:05d}.png"))
                    img_count += 1

    return exp_dir


if __name__ == "__main__":
    args, config = parse_args_and_config()
    exp_dir = main(args, config)
    print(exp_dir, end="")
