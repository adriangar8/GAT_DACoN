#!/bin/bash
#SBATCH --job-name=train_gat_dacon
#SBATCH --output=out_err/%x_%j.out
#SBATCH --error=out_err/%x_%j.err
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adrian.garcia@telecom-paris.fr

source /home/ids/adgarcia/miniconda3/etc/profile.d/conda.sh
conda activate dacon

echo "Conda env: $CONDA_DEFAULT_ENV"

python dacon/train.py \
    --config configs/train_gat.yaml \
    --version 1_1

echo "Job completed at $(date)"
