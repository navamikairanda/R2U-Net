#!/bin/bash
#SBATCH -p gpu20
#SBATCH -o /HPS/Navami/work/code/nnti/R2U-Net/logs/slurm-output/slurm-%j.out
#SBATCH -t 0-04:00:00
#SBATCH --gres gpu:4

cd /HPS/Navami/work/code/nnti/R2U-Net
#sbatch scripts/slurm_run.sh

## RUN
# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate nnti

#python -u main.py logs/expt1 #FCN
#python -u main.py logs/expt2 #U-Net
#python -u main.py logs/expt3 #R2U-Net (t=2)
#python -u main.py logs/expt4 #R2U-Net (t=3)
#python -u main.py logs/expt5 #Recurrent U-Net
#python -u main.py logs/expt6 #Residual U-Net
python -u main.py logs/expt7_2 #DeepLab V3