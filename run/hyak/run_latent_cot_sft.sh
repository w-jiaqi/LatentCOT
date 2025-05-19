#!/bin/bash
#SBATCH --job-name=gsm8k_latent_sft
#SBATCH --partition=gpu-l40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiaqiw18@uw.edu
#SBATCH --output=/mmfs1/gscratch/ark/jiaqi/projects/LatentCOT/logs/%j.out
#SBATCH --error=/mmfs1/gscratch/ark/jiaqi/projects/LatentCOT/logs/%j.err

source /mmfs1/gscratch/ark/jiaqi/miniconda3/etc/profile.d/conda.sh
conda activate latentcot
cd /mmfs1/gscratch/ark/jiaqi/projects/LatentCOT
python sft/latent_cot_sft.py -c $1

# usage: sbatch run/hyak/run_cot_sft.sh configs/latent_cot_sft/gsm8k/pool_8.yaml