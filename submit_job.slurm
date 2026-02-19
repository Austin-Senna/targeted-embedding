#!/bin/bash
#SBATCH --account=bgbh-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --job-name=a40_batch_job
#SBATCH --output=job_output_%j.txt

# --- Your environment setup and commands go below this line ---

# Load any necessary modules (example)
module load  miniforge3-python
eval "$(mamba shell hook --shell bash)"
mamba activate ./qwen_env

# Run your script (example)
python qwen_parallel_with_config.py