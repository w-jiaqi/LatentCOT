#!/bin/bash
#SBATCH --nodelist=gpu006
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --job-name=latent_sft_4x4
#SBATCH --output=/lustre/fs0/scratch/pkeung/multiplication/LatentCOT/logs/%x_%j/log.out

srun --container-image=/lustre/fs0/scratch/pkeung/squash/nvidia+pytorch+24.10-py3.sqsh \
     --container-mounts=/lustre/fs0 \
     bash -c "cd /lustre/fs0/scratch/pkeung/multiplication/LatentCOT && \
			 source env/bin/activate && \
             python sft/latent_cot_sft.py -d gsm8k -l 4 --batch_num 8 --checkpoints_name pool_4 --epochs 5

# usage: sbatch run_sft_latent_gsm8k_dgx.sh