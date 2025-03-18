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
parser.add_argument("-c", "--finetune_dir", type=str, default=None)
parser.add_argument("-d", "--device", default="cuda")
parser.add_argument("--dtype", default=torch.bfloat16)
parser.add_argument("--log_dir", default="eval/logs/multiplication")

args = parser.parse_args()

if args.finetune_dir == None:
	print("USING BASE MODEL")

base_model = AutoModelForCausalLM.from_pretrained(args.base_model)

model = PeftModel.from_pretrained(base_model, args.finetune_dir)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(args.finetune_dir)

generator = pipeline(
	"text-generation",
	model=model,
	tokenizer=tokenizer,
	torch_dtype=args.dtype,
	device_map=args.device
)


ds = dataset.get_4x4_multiplication_dataset(tokenizer, chat_template=False, eval_only=True)

import tqdm

def get_ans_from_response(response):
	answer = answer.split("####")[-1].strip().replace(" ", "")

	try:
		return int(answer)
	except ValueError:
		return answer

pb = tqdm(range(len(ds)))

correct = 0

for idx, example in enumerate(ds):
	pred_string = generator(example["prompt"], max_new_tokens=512)[0]['generated_text'][-1]['content']

	pred_ans = get_ans_from_response(pred_string)
	true_ans = get_ans_from_response(example['completion']['content'])

	if pred_ans == true_ans:
		correct += 1

	accuracy = (correct / (idx + 1)) * 100

	pb.set_description(print(f"{accuracy}%"))
	pb.update(1)


