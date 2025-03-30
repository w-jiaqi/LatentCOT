'''

test file for chatting with model

'''
import sys, os
sys.path.insert(0, os.path.abspath("."))  # hack for imports

from transformers import pipeline
import torch

import argparse

torch.set_default_device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model_path", type=str, required=True,
)
parser.add_argument(
    "-t", "--tokenizer_path", type=str, required=True,
)
parser.add_argument(
	"--max_new_tokens", type=int, default=256
)

args = parser.parse_args()

generate = pipeline(
    "text-generation", 
    model=args.model_path, 
	tokenizer=args.tokenizer_path,
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

while True:
	prompt = input()

	print(generate(prompt + '\n', 
				max_new_tokens=args.max_new_tokens))