'''

test file for chatting with the latent model

'''
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
    "-m", "--model_path", type=str, required=True,
)

parser.add_argument(
    "-l", "--latent_pool", type=int, required=True,
)

args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

start_latent_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
end_latent_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
start_cot_id = tokenizer.convert_tokens_to_ids("<|start-cot|>")
end_cot_id = tokenizer.convert_tokens_to_ids("<|end-cot|>")

while True:
	prompt = input()
	input_ids = tokenizer(prompt, return_tensors='pt').input_ids

	embedding = model.get_input_embeddings()
	inputs_embeds = (torch.cat((
		embedding(torch.tensor(start_latent_id)).unsqueeze(0), 
		compress_embeddings(embedding(input_ids)[0], args.latent_pool)[1],
		embedding(torch.tensor(end_latent_id)).unsqueeze(0),
		embedding(torch.tensor(start_cot_id)).unsqueeze(0)
	))).unsqueeze(0)
	
	attention_mask = torch.ones(inputs_embeds.shape[:-1])

	print(inputs_embeds.shape)
	print(attention_mask.shape)

	output = model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=256)
	generated_text = tokenizer.decode(output[0])

	print(generated_text)