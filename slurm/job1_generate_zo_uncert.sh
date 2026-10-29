#!/bin/bash
#SBATCH --job-name=uvit_zo_uncert
#SBATCH --output=logs/generate/uvit_%j.out
#SBATCH --error=logs/generate/uvit_%j.err
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --array=0-4

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH

# General variables
sample_batch_size="25"
total_n_sample="10000"
fixed_class="10000"
DEVICES=0
seed=1234
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}

# Parameters
steps="50"
DIS="time_uniform"
mc_size="5"

mc_steps="10"
sigma_perturb="0.125"

zo_iters="3"
zo_neighborhood="2"
zo_lambda="0.2"

# Paths
path="imagenet256_uvit_huge.py"
encoder_path="checkpoints/stable-diffusion/autoencoder_kl_ema.pth"
uvit_path="checkpoints/imagenet256_uvit_huge.pth"
exp_path="UVIT_sample_results/zo_uncert"

echo "Running for seed $seed"

cd UViT

python main_zo_uncert.py \
  --config $path \
  --device 0 \
  --mc_timesteps $mc_steps \
  --timesteps $steps \
  --skip_type $DIS \
  --mc_size $mc_size \
  --sigma_perturb $sigma_perturb \
  --fixed_class $fixed_class \
  --sample_batch_size $sample_batch_size \
  --total_n_sample $total_n_sample \
  --encoder_path $encoder_path \
  --uvit_path $uvit_path \
  --seed $seed \
  --exp_path $exp_path \
  --zo_iters $zo_iters \
  --zo_neighborhood $zo_neighborhood \
  --zo_lambda $zo_lambda \
  --num_shards $NUM_SHARDS \
  --shard_id $SHARD_ID
