#!/bin/bash

#SBATCH --job-name=gpu
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=00:00:00
#SBATCH --nodes=1
#SBATCH --partition=LocalQ  # Local partition
#SBATCH --ntasks=1        # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1        # Request 5 GPU
# Add lines here to run your GPU-based computations.
source /home/jjc/anaconda3/etc/profile.d/conda.sh
conda activate llm_quzo
cd /home/jjc/project/QuZO/large_models/script/
bash mistral_quzo_exp_lora.sh