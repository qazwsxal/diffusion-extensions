#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1

# SBATCH --exclude="gpu[0-6]"
#SBATCH -p "res-gpu-small"
#SBATCH --qos="short"
#SBATCH -t 2-00

# Source the bash profile (required to use the module command)
source /etc/profile
module unload cuda
module load cuda/11.1

source .venv/bin/activate
export PYTORCH_JIT=0
# export CUDA_LAUNCH_BLOCKING=1
python3 -u prot_train.py