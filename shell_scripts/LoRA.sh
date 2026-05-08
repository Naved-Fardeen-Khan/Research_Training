#!/bin/bash
#SBATCH --job-name=LoRA_final_run     # Job name
#SBATCH --account=project_2018500     # Project number
#SBATCH --partition=gpu
#SBATCH --time=72:00:00               # Maximum runtime (3 days)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10            # Use more CPU cores for data loading
#SBATCH --mem=64G                     # More than enough for a quick test
#SBATCH --gres=gpu:v100:1             # 1 V100 GPU
#SBATCH --output=LoRA_finetune_log_%j.txt  # Nicely names output file

module load pytorch

# Routes the heavy model weights to scratch
export HF_HOME=/scratch/project_2018500/hf_cache  
# Routes the formatted datasets to scratch so your home dir doesn't explode!
export HF_DATASETS_CACHE=/scratch/project_2018500/hf_cache/datasets 

# Launch the script!
srun python3 LoRA.py

