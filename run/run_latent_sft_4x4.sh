#!/bin/bash
#SBATCH --nodelist=gpu006
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --job-name=latent_sft_4x4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
HF_TOKEN=$1

srun --container-image=/lustre/fs0/scratch/pkeung/squash/nvidia+pytorch+24.10-py3.sqsh \
     --container-mounts=/lustre/fs0 \
     bash -c "cd /lustre/fs0/scratch/pkeung/multiplication/LatentCOT && \
			 source env/bin/activate && \
             export HF_HOME=$PWD/hf_cache && \
             export HF_TOKEN=$HF_TOKEN && \
             export TRANSFORMERS_CACHE=$PWD/hf_cache/transformers && \
             export HF_DATASETS_CACHE=$PWD/hf_cache/datasets && \
             python sft/latent_cot_sft.py -l 4 -d 4x4"

# usage: sbatch run_latent_sft_4x4.sh hf_token