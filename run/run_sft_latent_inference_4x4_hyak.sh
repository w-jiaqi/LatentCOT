#!/bin/bash
#SBATCH --job-name=4x4_cold_start
#SBATCH --partition=gpu-l40s
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anjo0@uw.edu
#SBATCH --output=/mmfs1/gscratch/ark/anjo0/LatentCOT/logs/%x_%j/log.out

cd /gscratch/ark/anjo0/LatentCOT
source env/bin/activate
python sft/latent_cot_grpo.py -d 4x4 --checkpoints_name cold_start --max_new_latents 8 --batch_num 16 --epochs 5
