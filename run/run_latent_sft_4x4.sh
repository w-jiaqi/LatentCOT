#!/bin/bash
#SBATCH --job-name=latent_sft_4x4
#SBATCH --partition=gpu-l40s
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
export HF_TOKEN=
python sft/latent_cot_sft.py --dataset 4x4 --model meta-llama/Llama-3.2-1B