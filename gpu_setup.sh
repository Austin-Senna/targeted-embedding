srun --account=bgbh-delta-gpu --partition=gpuA100x4-interactive --nodes=1 --gpus-per-node=1 --tasks=1 --cpus-per-task=16 --mem=64g --time=01:00:00 --pty /bin/bash

# to get all 
squeue --me
ssh 'instance-name'

module load  miniforge3-python
eval "$(mamba shell hook --shell bash)"
mamba activate ./qwen_env

# for tts
# use python 3.10 for tts, otherwise it is hard
# module load  miniforge3-python
# eval "$(mamba shell hook --shell bash)"
# mamba activate ./tts_env
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH



# to start a new training job that can be detached
tmux new -s training_job
tmux attach -t training_job