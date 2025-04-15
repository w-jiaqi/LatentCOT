#!/bin/bash
#SBATCH --job-name=sft_latent_4x4_pool_8
#SBATCH --partition=gpu-l40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anjo0@uw.edu
#SBATCH --output=/mmfs1/gscratch/ark/anjo0/LatentCOT/logs/%j/log.out
#SBATCH --error=/mmfs1/gscratch/ark/anjo0/LatentCOT/logs/%j/log.err

cd /gscratch/ark/anjo0/LatentCOT
source env/bin/activate
python sft/latent_cot_sft.py -d 4x4 -l 8 --batch_num 4 --checkpoints_name cot_and_ans