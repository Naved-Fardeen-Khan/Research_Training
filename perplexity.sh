#!/bin/bash
#SBATCH --job-name=test_phi3
#SBATCH --account=project_2018500
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --output=ppl_test_log_%j.txt  # Nicely names output file

module load pytorch

# Routes the heavy model weights to scratch
export HF_HOME=/scratch/project_2018500/hf_cache  
# Routes the formatted datasets to scratch so your home dir doesn't explode!
export HF_DATASETS_CACHE=/scratch/project_2018500/hf_cache/datasets 

# Launch the script!
python3 perplexity.py