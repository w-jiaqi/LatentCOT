#!/bin/bash
#SBATCH --nodelist=gpu006
#SBATCH --time=96:00:00
#SBATCH --gpus=1
#SBATCH --job-name=latent_freeform
#SBATCH --output=/lustre/fs0/scratch/pkeung/multiplication/LatentCOT/logs/%x_%j/log.out

srun --container-image=/lustre/fs0/scratch/pkeung/squash/nvidia+pytorch+24.10-py3.sqsh \
     --container-mounts=/lustre/fs0 \
     bash -c "cd /lustre/fs0/scratch/pkeung/$1/LatentCOT && \
               source env/bin/activate && \
               python sft/latent_cot_freeform.py -c $2"

# usage: sbatch run_latent_freeform_dgx.sh (multiplication | gsm8k) (config_path)