import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
	pipeline
)
from peft import PeftModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument("-c", "--checkpoint_dir", type=str, required=True)
parser.add_argument("-d", "--device", default="cuda")
parser.add_argument("--dtype", default=torch.bfloat16)

args = parser.parse_args()

base_model = AutoModelForCausalLM.from_pretrained(args.base_model)

model = PeftModel.from_pretrained(base_model, args.checkpoint_dir)
model = model.merge_and_unload()

generator = pipeline(
	"text-generation",
	model=args.base_model,
	torch_dtype=args.dtype,
	device_map=args.device
)

messages = [
	{"role": "user", "content": "Question: 5 6 3 2 * 7 4 3 4"},
]

outputs = generator(
	messages,
	max_new_tokens=256
)

print(outputs[0]["generated_text"][-1])