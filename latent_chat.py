'''

test file for chatting with the latent model

'''
import sys, os

sys.path.insert(0, os.path.abspath("."))  # hack for imports
import torch

from sft.models.latent_cot_model import LatentCOTModel
from sft.models.latent_tokenizer import LatentTokenizer
from utils.utils import get_config

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == 'cpu':
        print("WARNING: USING CPU")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--config", type=str, required=True,
)

args = parser.parse_args()

config = get_config(args.config)

tokenizer = LatentTokenizer(config.tokenizer)
model = LatentCOTModel(config.base_model, tokenizer, freeze_embeddings=True).to(device)

if config.model_pth is not None:
    model.load_state_dict(torch.load(config.model_pth))

model.eval()

print("Latent Chat Ready")

while True:
	prompt = input()

	tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

	cot_ids = model.generate(
              	inputs_ids=tokens['input_ids'], 
				input_attention_mask=tokens['attention_mask'], 
				max_new_latents=config.max_new_latents, 
				max_new_tokens=256, 
				probe_latents=config.probe_latents, 
				output_cot=True, 
				unembed_latents=config.unembed_latents, 
				dynamically_stop=config.dynamically_stop)

	ans_ids = model.generate(
				inputs_ids=tokens['input_ids'], 
				input_attention_mask=tokens['attention_mask'], 
				max_new_latents=config.max_new_latents, 
				max_new_tokens=256, 
				output_cot=False,
				unembed_latents=config.unembed_latents,
				dynamically_stop=config.dynamically_stop)

	print(cot_ids)
	print(tokenizer.decode(cot_ids[0]))

	print(ans_ids)
	print(tokenizer.decode(ans_ids[0]))