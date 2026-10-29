#!/bin/bash
#SBATCH --job-name=semantic_likelihood
#SBATCH --output=logs/uncertainty/eval_%j.out
#SBATCH --error=logs/uncertainty/eval_%j.err
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=00:30:00

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH

# Adjustable variables
path="UViT/UVIT_sample_results/laplace/imagenet256/dpmUQ_fixed_class10000_train%50_step50_S5_epi_unc_1234"

python -u semantic_likelihood.py --path=$path
