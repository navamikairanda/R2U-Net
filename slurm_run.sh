#!/bin/bash
#SBATCH -p gpu20
#SBATCH -o /HPS/Navami/work/code/nnti/Vision-Project-Image-Segmentation/slurm-output/slurm-%j.out
#SBATCH -t 0-04:00:00
#SBATCH --gres gpu:2

cd /HPS/Navami/work/code/nnti/Vision-Project-Image-Segmentation
#sbatch slurm_run.sh

## RUN
# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate nnti

python -u task_1.py logs/expt1

