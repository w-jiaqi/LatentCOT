#!/bin/bash
#SBATCH --job-name=llama3_gsm8k_training
#SBATCH --partition=gpu-a40
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

conda activate latentcot
cd /gscratch/ark/jiaqi/llama3_gsm8k
python fine_tune_llama3_gsm8k.py