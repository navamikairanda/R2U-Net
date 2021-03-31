#!/bin/bash
#SBATCH -p gpu20
#SBATCH -o /HPS/Navami/work/code/nnti/R2U-Net/logs/slurm-output/slurm-%j.out
#SBATCH -t 0-08:00:00
#SBATCH --gres gpu:2

cd /HPS/Navami/work/code/nnti/R2U-Net
#sbatch scripts/slurm_run.sh

## RUN
# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate nnti

#python -u main.py logs/expt1_0 #FCN, bs=16
#python -u main.py logs/expt2_0 #U-Net, bs=16
#python -u main.py logs/expt3_0 #R2U-Net (t=2), bs=16
#python -u main.py logs/expt4_0 #R2U-Net (t=3), bs=8
#python -u main.py logs/expt5_0 #Recurrent U-Net
#python -u main.py logs/expt6_0 #Residual U-Net
#python -u main.py logs/expt7_0 #DeepLab V3 VGG backbone, bs=16
#python -u main.py logs/expt8_0 #DeepLab V3 Resnet backbone, bs=16
python -u main.py logs/expt8_1 #DeepLab V3 Resnet backbone, bs=16, LR decay