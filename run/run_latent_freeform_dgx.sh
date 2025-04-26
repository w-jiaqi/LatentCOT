#!/bin/bash
#SBATCH --nodelist=gpu006
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --job-name=latent_freeform
#SBATCH --output=/lustre/fs0/scratch/pkeung/multiplication/LatentCOT/logs/%x_%j/log.out

srun --container-image=/lustre/fs0/scratch/pkeung/squash/nvidia+pytorch+24.10-py3.sqsh \
     --container-mounts=/lustre/fs0 \
     bash -c "cd /lustre/fs0/scratch/pkeung/multiplication/LatentCOT && \
               source env/bin/activate && \
               python sft/latent_cot_grpo.py -c $1"