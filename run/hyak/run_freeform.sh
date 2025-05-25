#!/bin/bash
#SBATCH --job-name=freeform_sft
#SBATCH --partition=gpu-l40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anjo0@uw.edu
#SBATCH --output=/gscratch/ark/anjo0/LatentCOT/run/logs/%j_%x.out

cd /gscratch/ark/anjo0/LatentCOT
source env/bin/activate
python sft/latent_cot_freeform.py -c $1

# Note: this script will only work for multiplication (need to change mail, output and cd argument)