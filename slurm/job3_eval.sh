#!/bin/bash
#SBATCH --job-name=metrics_eval
#SBATCH --output=logs/eval/metrics_%j.out
#SBATCH --error=logs/eval/metrics_%j.err
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

# Fixed variables
H=256
m=0
unc_name="entropy_clip"
reverse="false"

# Adjustable variables
EXP="../UViT/UVIT_sample_results/laplace/imagenet256/dpmUQ_fixed_class10000_train%50_step50_S5_epi_unc_1234"

cd eval

for NUM in 6000 7000 8000 9000 10000 11000 12000
do
    echo "Computing FID, precision, recall for Random baseline (N=$NUM)"
    python fid.py calc --images="${EXP}/${m}/imgs" --ref=./fid-refs/imagenet-${H}x${H}.npz --num $NUM --fid_features="${EXP}/${m}/fid_features.pt"
    python precision_recall_torch.py --ref ./precision-recall-refs/image_net_${H}_fid_features_.pt --eval "${EXP}/${m}/fid_features.pt"

    echo "Computing FID, precision, recall for filtered samples based on generative uncertainty (N=$NUM)"
    python idx_sort.py --path ${EXP} --name ${unc_name} --N $NUM --reverse ${reverse}
    python fid.py calc --images="${EXP}/${m}/imgs" --ref=./fid-refs/imagenet-${H}x${H}.npz --fid_features="${EXP}/${m}/fid_features_filtered_${unc_name}.pt" --idx_path="${EXP}/idx_sorted_${NUM}_${unc_name}.npy"
    python precision_recall_torch.py --ref ./precision-recall-refs/image_net_${H}_fid_features_.pt --eval "${EXP}/${m}/fid_features_filtered_${unc_name}.pt"

    echo "Done with NUM = $NUM"
    echo ""
done
