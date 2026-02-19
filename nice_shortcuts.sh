# Expensive, but is interactive
srun --account=bgbh-delta-gpu --partition=gpuA100x4-interactive --nodes=1 --gpus-per-node=1 --tasks=1 --cpus-per-task=16 --mem=64g --time=01:00:00 --pty /bin/bash

# sbatch submit_job.slurm, rewrite the file yourself
sbatch submit_job.slurm

# to get all 
squeue --me
ssh 'instance-name'

# python environment setup
module load  miniforge3-python
eval "$(mamba shell hook --shell bash)"
mamba activate ./qwen_env

# to start a new training job that can be detached
tmux new -s training_job
tmux attach -t training_job