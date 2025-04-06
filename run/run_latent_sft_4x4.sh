#!/bin/bash
#SBATCH --nodelist=gpu006
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --job-name=latent_sft_4x4
#SBATCH --output=%x_%j.out
HF_TOKEN=$1

srun --container-image=/lustre/fs0/scratch/pkeung/squash/nvidia+pytorch+24.10-py3.sqsh \
     --container-mounts=/lustre/fs0 \
     bash -c "cd /lustre/fs0/scratch/pkeung/multiplication/LatentCOT && \
			 source env/bin/activate && \
             python sft/latent_cot_sft.py -d 4x4 -l 4 -m meta-llama/Llama-3.2-3B -t meta-llama/Llama-3.2-3B --batch_num 8 --lr 1e-4 --num_train 300000 --checkpoints_name 3B_300k --no_cache"

# usage: sbatch run_latent_sft_4x4.sh