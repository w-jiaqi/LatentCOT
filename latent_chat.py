'''

test file for chatting with the latent model

'''
import sys, os

from data.dataset import compress_embeddings
from utils import utils
sys.path.insert(0, os.path.abspath("."))  # hack for imports

from sft.models.latent_cot_model import LatentCOTModel

import torch

from sft.models.latent_tokenizer import LatentTokenizer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model", type=str, required=True,
)

parser.add_argument(
	"-t", "--tokenizer", type=str, required=True
)

parser.add_argument(
	"--tie_weights", action='store_true'
)

args = parser.parse_args()

tokenizer = LatentTokenizer(args.tokenizer)
model = LatentCOTModel(args.model, tokenizer, tie_weights=args.tie_weights).to('cuda')

print("Latent Chat Ready")

while True:
	prompt = input()

	prompt_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)[0].to('cuda')

	print(model.generate(prompt_ids, max_new_latents=21, max_new_tokens=256, probe_latents=True, output_cot=False))
	print(model.generate(prompt_ids, max_new_latents=21, max_new_tokens=256, probe_latents=True, output_cot=True))