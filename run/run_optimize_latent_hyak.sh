#!/bin/bash
#SBATCH --job-name=optimize_latent
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

export HF_TOKEN=1

source /mmfs1/gscratch/ark/jiaqi/miniconda3/etc/profile.d/conda.sh
conda activate latentcot
cd /mmfs1/gscratch/ark/jiaqi/projects/LatentCOT

python optimize_latents.py \
  --dataset gsm8k \
  --model /mmfs1/gscratch/ark/jiaqi/projects/latent/1B_3k/model \
  --tokenizer /mmfs1/gscratch/ark/jiaqi/projects/latent/1B_3k/tokenizer \
  --latent_pool 4 \
  --steps 100 \
  --lr 1e-3 \
  --batch_size 8 \
  --output_dir /mmfs1/gscratch/ark/jiaqi/projects/LatentCOT/optimized_latents \
  --log_every 20 \
  --test_examples 5