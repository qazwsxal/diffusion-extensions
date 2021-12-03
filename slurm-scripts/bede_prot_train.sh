#!/bin/bash

# Generic options:

#SBATCH --account=bddur05  # Run job under project <project>
#SBATCH --time=2-0:00:00      # Run for a max of 2 days

# Node resources:

#SBATCH --partition=gpu    # Choose either "gpu" or "infer" node type
#SBATCH --nodes=1
#SBATCH --gres=gpu:1    # 25% of node CPU and RAM per GPU


PYTHON_VIRTUAL_ENVIRONMENT=diffusion_ext
CONDA_ROOT=/users/wpzx47/miniconda3

## Activate virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

python3 -u prot_train.py --batch=2