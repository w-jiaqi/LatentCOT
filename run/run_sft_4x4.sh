#!/bin/bash
#SBATCH --job-name=sft_4x4
#SBATCH --partition=gpu-a40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anjo0@uw.edu
#SBATCH --output=/mmfs1/gscratch/ark/anjo0/LatentCOT/logs/%j.out
#SBATCH --error=/mmfs1/gscratch/ark/anjo0/LatentCOT/logs/%j.err

source /mmfs1/gscratch/ark/anjo0/miniconda3/etc/profile.d/conda.sh
conda activate latentcot
cd /mmfs1/gscratch/ark/jiaqi/projects/LatentCOT
python sft/cot_sft.py -d 4x4