'''

test file for chatting with the latent model

'''
import sys, os

sys.path.insert(0, os.path.abspath("."))  # hack for imports
import torch

from sft.models.latent_cot_model import LatentCOTModel
from sft.models.latent_tokenizer import LatentTokenizer

import argparse
import yaml
from types import SimpleNamespace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == 'cpu':
        print("WARNING: USING CPU")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--config", type=str, required=True,
)

args = parser.parse_args()

with open(args.config, "r") as f:
	config = SimpleNamespace(**yaml.safe_load(f))

tokenizer = LatentTokenizer(config.tokenizer)
model = LatentCOTModel(config.base_model, tokenizer, freeze_embeddings=True).to(device)

if config.model_pth is not None:
    model.load_state_dict(torch.load(config.model_pth))

model.eval()

print("Latent Chat Ready")

while True:
	prompt = input()

	tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to('cuda')

	print(model.generate(tokens['input_ids'], tokens['attention_mask'], max_new_latents=config.max_new_latents, max_new_tokens=256, probe_latents=config.probe_latents, output_cot=True))
	print(model.generate(tokens['input_ids'], tokens['attention_mask'], max_new_latents=config.max_new_latents, max_new_tokens=256, output_cot=False))