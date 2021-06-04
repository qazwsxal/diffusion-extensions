#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1


#SBATCH -p "res-gpu-small"
#SBATCH --qos="debug"
#SBATCH -t 0-02

# Source the bash profile (required to use the module command)
source /etc/profile
module unload cuda
module load cuda/10.2-cudnn8.1

source .venv/bin/activate

python3 -u jigsaw_translate.py