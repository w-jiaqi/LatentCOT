#!/bin/bash
#SBATCH --nodelist=gpu006
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --job-name=optimize_latents
#SBATCH --output=%x_%j.out

export HF_TOKEN=1

srun --container-image=/lustre/fs0/scratch/pkeung/squash/nvidia+pytorch+24.10-py3.sqsh \
     --container-mounts=/lustre/fs0 \
     bash -c "
         cd /lustre/fs0/scratch/pkeung/gsm8k/LatentCOT && \
         source env/bin/activate && \
         python optimize_latents.py \
         --dataset gsm8k \
         --model /lustre/fs0/scratch/pkeung/gsm8k/LatentCOT/checkpoints/latent-cot-sft/gsm8k/1B_3k/model \
         --tokenizer /lustre/fs0/scratch/pkeung/gsm8k/LatentCOT/checkpoints/latent-cot-sft/gsm8k/1B_3k/tokenizer \
         --latent_pool 4 \
         --steps 100 \
         --lr 1e-3 \
         --output_dir /lustre/fs0/scratch/pkeung/gsm8k/LatentCOT/optimized_latents \
         --log_every 20 \
         --test_examples 5
     "