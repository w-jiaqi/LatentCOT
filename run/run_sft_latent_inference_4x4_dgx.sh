#!/bin/bash
#SBATCH --nodelist=gpu006
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --job-name=4x4_cold_start
#SBATCH --output=/lustre/fs0/scratch/pkeung/multiplication/LatentCOT/logs/%x_%j/log.out

srun --container-image=/lustre/fs0/scratch/pkeung/squash/nvidia+pytorch+24.10-py3.sqsh \
     --container-mounts=/lustre/fs0 \
     bash -c "cd /lustre/fs0/scratch/pkeung/multiplication/LatentCOT && \
               source env/bin/activate && \
               python sft/latent_cot_grpo.py -d 4x4 -e 5 --batch_num 32 --max_new_latents 8 --checkpoints_name cold_start"
# usage: sbatch run_sft_latent_gsm8k_dgx.sh