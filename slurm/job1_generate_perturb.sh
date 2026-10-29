#!/bin/bash
#SBATCH --job-name=uvit_perturb
#SBATCH --output=logs/generate/uvit_%j.out
#SBATCH --error=logs/generate/uvit_%j.err
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=10:00:00

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH

# General variables
sample_batch_size="128"
total_n_sample="12032"
fixed_class="10000"
DEVICES=0
seed=1234

# Parameters
steps="50"
DIS="time_uniform"
mc_size="5"

mc_steps="50"
sigma_perturb="0.125"

# Paths
path="imagenet256_uvit_huge.py"
encoder_path="checkpoints/stable-diffusion/autoencoder_kl_ema.pth"
uvit_path="checkpoints/imagenet256_uvit_huge.pth"
exp_path="UVIT_sample_results/perturb"

echo "Running for seed $seed"

cd UViT

CUDA_VISIBLE_DEVICES=$DEVICES python main_perturb.py \
  --config $path \
  --timesteps $steps \
  --eta 0 \
  --skip_type $DIS \
  --mc_size $mc_size \
  --sample_batch_size $sample_batch_size \
  --fixed_class $fixed_class \
  --total_n_sample $total_n_sample \
  --encoder_path $encoder_path \
  --uvit_path $uvit_path \
  --seed $seed \
  --exp_path $exp_path \
  --mc_timesteps $mc_steps \
  --sigma_perturb $sigma_perturb
