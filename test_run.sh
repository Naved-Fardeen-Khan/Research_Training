#!/bin/bash
#SBATCH --job-name=test_run
#SBATCH --account=project_2018500     # Project number
#SBATCH --partition=gpu
#SBATCH --time=00:15:00               # 15 minutes for a quick test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # Don't need as many CPUs just for test
#SBATCH --mem=32G                     # 32GB RAM is plenty
#SBATCH --gres=gpu:v100:1             # 1 V100 GPU

module load pytorch

# Crucial: Route the download to scratch cache
export HF_HOME=/scratch/project_2018500/hf_cache

# Run the test
srun python3 test_run.py
