'''

test file for chatting with the latent model

'''
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import argparse

torch.set_default_device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model_path", type=str, required=True,
)

args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

# start_latent_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
# start_latent_id_tensor = torch.tensor([[start_latent_id]])

while True:
	prompt = input()
	input_ids = tokenizer(prompt, return_tensors='pt').input_ids
	attention_mask = torch.ones_like(input_ids)

	output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50)
	generated_text = tokenizer.decode(output[0])

	print(output[0])
	print(generated_text)