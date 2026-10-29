# UViT experiments
Code here is based on the following 2 repos:
- [BayesDiff](https://github.com/karrykkk/BayesDiff)
- [UViT](https://github.com/baofff/U-ViT)

## Setup
- Download the UViT model checkpoint using [this link](https://drive.google.com/file/d/13StUdrjaaSXjfqqF7M47BzPyhMAArQ4u/view?usp=sharing) and save it as `checkpoints/imagenet256_uvit_huge.pth`
- Download the Stable Diffusion autoencoder checkpoint using [this link](https://drive.google.com/drive/folders/1yo-XhqbPue3rp5P57j6QbA5QZx6KybvP?usp=sharing) and save it as `checkpoints/stable-diffusion/autoencoder_kl_ema.pth`
- Download the [ImageNet dataset](https://www.image-net.org/) and update `config.dataset.path` in `./configs/imagenet256_uvit_huge.py` (only used by `main_laplace.py` to fit the Laplace posterior)
- Complete the remaining **Setup** steps in the [root README](../README.md#setup)
