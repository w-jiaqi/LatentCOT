'''

test file for chatting with the latent model

'''
import sys, os
sys.path.insert(0, os.path.abspath("."))  # hack for imports

from sft.models.latent_2_text import Latent2Text
from sft.models.text_2_latent import Text2Latent

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

import argparse
from data.dataset import compress_embeddings

torch.set_default_device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument(
    "-e", "--text_to_latent", type=str, required=True,
)

parser.add_argument(
    "-d", "--latent_to_text", type=str, required=True,
)

parser.add_argument(
    "-l", "--latent_pool", type=int, required=True,
)

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("checkpoints/latent-cot-sft/4x4/latent_to_text")

text_to_latent = Text2Latent(model_id=args.text_to_latent, tokenizer=tokenizer)
latent_to_text = Latent2Text(model_id=args.latent_to_text, tokenizer=tokenizer)

start_latent_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
end_latent_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
start_cot_id = tokenizer.convert_tokens_to_ids("<|start-cot|>")
end_cot_id = tokenizer.convert_tokens_to_ids("<|end-cot|>")

while True:
	prompt = input()

	print(latent_to_text.generate(
		text_to_latent.generate(
			prompt, 
			max_new_embeds=20, 
			start_latent=start_latent_id, 
			end_latent=end_latent_id), 
        start_cot_id=None
    ))
	