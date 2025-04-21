#!/bin/bash
#SBATCH --job-name=sft_latent_inference_4x4
#SBATCH --partition=gpu-l40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anjo0@uw.edu
#SBATCH --output=/mmfs1/gscratch/ark/anjo0/LatentCOT/logs/%j/log.out
#SBATCH --error=/mmfs1/gscratch/ark/anjo0/LatentCOT/logs/%j/log.err

cd /gscratch/ark/anjo0/LatentCOT
source env/bin/activate
python sft/latent_cot_grpo.py -d 4x4 --model checkpoints/latent-cot-sft/4x4/pool_8/model -t checkpoints/latent-cot-sft/4x4/pool_8/tokenizer --checkpoints_name pool_8
