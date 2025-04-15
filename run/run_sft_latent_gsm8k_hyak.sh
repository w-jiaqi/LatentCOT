#!/bin/bash
#SBATCH --job-name=sft_latent_gsm8k_pool_4
#SBATCH --partition=gpu-a100
#SBATCH --account=cse
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
python sft/latent_cot_sft.py -d gsm8k -l 8 --batch_num 4 --epochs 25