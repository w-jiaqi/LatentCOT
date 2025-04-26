'''

test file for chatting with the latent model

'''
import sys, os

from data.dataset import compress_embeddings
from utils import utils
sys.path.insert(0, os.path.abspath("."))  # hack for imports

from sft.models.latent_cot_model import LatentCOTModel

import torch

from sft.models.latent_tokenizer import LatentTokenizer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model", type=str, required=True,
)

parser.add_argument(
	"-t", "--tokenizer", type=str, required=True
)

parser.add_argument(
	"--tie_weights", action='store_true'
)

parser.add_argument(
	"-l", "--latent_pool", type=int, required=True
)
parser.add_argument(
	"--max_new_latents", type=int, required=True
)
parser.add_argument(
        "-p", "--pth", type=str
)

args = parser.parse_args()

tokenizer = LatentTokenizer(args.tokenizer)
model = LatentCOTModel(args.model, tokenizer, tie_weights=args.tie_weights).to('cuda')

if args.pth is not None:
    model.load_state_dict(torch.load(args.pth))

model.eval()

print("Latent Chat Ready")

while True:
	prompt = input()

	prompt_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)[0].to('cuda')

	print(model.generate(prompt_ids, max_new_latents=args.max_new_latents, max_new_tokens=256, probe_latents=True, output_cot=True))
	print(model.generate(prompt_ids, max_new_latents=args.max_new_latents, max_new_tokens=256, output_cot=False))

# torch.set_default_device('cuda')
# question = "5 6 3 2 * 7 4 3 4"
# # reasoning = "5 5 5 6 1 + 0 0 6 4 9 0 ( 5 5 1 1 1 1 ) + 0 0 5 9 0 7 0 ( 5 5 6 0 2 8 0 ) + 0 0 0 0 6 4 9 0"
# reasoning = "6 7 1 1 3 + 0 4 8 7 0 2 ( 6 1 0 9 3 2 ) + 0 0 4 8 7 0 2 ( 6 1 4 7 1 3 2 ) + 0 0 0 2 7 3 6 3"
# question_ids = tokenizer.encode(question, return_tensors="pt", add_special_tokens=False).to('cuda')[0]
# reasoning_ids = tokenizer.encode(reasoning, return_tensors="pt", add_special_tokens=False).to('cuda')[0]

# question_embeddings = model.embedding(question_ids)
# latent_reasoning_length, latent_reasoning_embeddings = compress_embeddings(model.embedding(reasoning_ids), 4)

# bos_col = torch.tensor(tokenizer.bos_token_id).unsqueeze(0).to('cuda')
# bos_col_embed = model.embedding(bos_col)

# start_latent_col = torch.tensor(tokenizer.start_latent_id).unsqueeze(0).to('cuda')
# start_latent_col_embed = model.embedding(start_latent_col)

# end_latent_col = torch.tensor(tokenizer.end_latent_id).unsqueeze(0).to('cuda')
# end_latent_col_embed = model.embedding(end_latent_col)

# start_cot_col = torch.tensor(tokenizer.start_cot_id).unsqueeze(0).to('cuda')
# start_cot_col_embed = model.embedding(start_cot_col)

# print(bos_col_embed.shape)
# print(question_embeddings.shape)
# print(latent_reasoning_embeddings.shape)
# print(start_latent_col_embed.shape)

# inputs_embeds = torch.cat((
# 	bos_col_embed,
# 	question_embeddings,
# 	start_latent_col_embed,
# 	latent_reasoning_embeddings,
# 	end_latent_col_embed,
# 	start_cot_col_embed
# ), dim=0).unsqueeze(0)

# attention_mask = torch.ones(inputs_embeds.shape[:-1])

# output = model.model.generate(
# 	inputs_embeds=inputs_embeds,
# 	attention_mask=attention_mask,
# 	max_new_tokens=256
# )

# print(tokenizer.decode(output[0]))
