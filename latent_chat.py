'''

test file for chatting with the latent model

'''
import sys, os
sys.path.insert(0, os.path.abspath("."))  # hack for imports

from sft.models.latent_2_text import Latent2Text
from sft.models.text_2_latent import Text2Latent

import torch

from sft.models.latent_tokenizer import LatentTokenizer

import argparse

torch.set_default_device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument(
    "-e", "--text_to_latent", type=str, required=True,
)

parser.add_argument(
    "-d", "--latent_to_text", type=str, required=True,
)

parser.add_argument(
	"-t", "--tokenizer", type=str, required=True
)

parser.add_argument(
    "-l", "--latent_pool", type=int, required=True,
)

args = parser.parse_args()

tokenizer = LatentTokenizer(args.tokenizer)

text_to_latent = Text2Latent(model_id=args.text_to_latent, tokenizer=tokenizer)
latent_to_text = Latent2Text(model_id=args.latent_to_text, tokenizer=tokenizer)

while True:
	prompt = input()

	print(latent_to_text.generate(
		text_to_latent.generate(
			prompt, 
			max_new_embeds=10), 
			output_cot=True
    ))

	print(latent_to_text.generate(
		text_to_latent.generate(
			prompt, 
			max_new_embeds=10), 
			output_cot=False
    ))

	print(text_to_latent.generate(prompt, max_new_embeds=10))
	print(text_to_latent.generate(prompt, max_new_embeds=10).shape)

	prompt_embedding = latent_to_text.embedding(tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False))
	prompt_embedding = torch.cat((
		latent_to_text.embedding(tokenizer.encode("<|start-latent|>", return_tensors="pt", add_special_tokens=True)),
		prompt_embedding,
		latent_to_text.embedding(tokenizer.encode("<|end-latent|>", return_tensors="pt", add_special_tokens=False)),
	), dim=1)

	print(latent_to_text.generate(prompt_embedding, output_cot=True))
	