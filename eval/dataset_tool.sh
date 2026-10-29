#!/bin/bash
#SBATCH --job-name=fid_refs
#SBATCH --output=../logs/dataset/dataset_%j.out
#SBATCH --error=../logs/dataset/dataset_%j.err
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=00:45:00

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH

source="/scratch-nvme/ml-datasets/imagenet/ILSVRC/Data/CLS-LOC/train"

python -u dataset_tool.py --source=$source --dest=./fid-refs/imagenet-256x256.zip --resolution=256x256 --transform=center-crop --max-images 50000
python -u fid.py ref --data=./fid-refs/imagenet-256x256.zip --dest=./fid-refs/imagenet-256x256.npz --fid_features=./precision-recall-refs/image_net_256_fid_features_.pt