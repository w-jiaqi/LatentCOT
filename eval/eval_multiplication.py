import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
	pipeline
)
from peft import PeftModel
import argparse

import sys, os
sys.path.insert(0, os.path.abspath('.')) # hack for imports

from data import dataset

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument("-c", "--checkpoint_dir", type=str, default=None)
parser.add_argument("-d", "--device", default="cuda")
parser.add_argument("--dtype", default=torch.bfloat16)

args = parser.parse_args()

if args.checkpoint_dir == None:
	print("USING BASE MODEL")

base_model = AutoModelForCausalLM.from_pretrained(args.base_model)

model = PeftModel.from_pretrained(base_model, args.checkpoint_dir)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)

generator = pipeline(
	"text-generation",
	model=model,
	tokenizer=tokenizer,
	torch_dtype=args.dtype,
	device_map=args.device
)


ds = dataset.get_4x4_multiplication_dataset(tokenizer, chat_template=False, eval_only=True)

import tqdm

# pb = tqdm()

correct = 0

for idx, example in enumerate(ds):
	print(example)
	print(generator(example["prompt"]))
