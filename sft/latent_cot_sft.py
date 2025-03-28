import sys, os
sys.path.insert(0, os.path.abspath("."))  # hack for imports

import torch
from torch.utils.data import DataLoader
import argparse
from utils import utils
from transformers import AutoTokenizer, AutoModelForCausalLM
from data.multiplication_dataset import get_text_to_latent_dataset, get_latent_to_text_dataset, collate_fn
from sft.models.text_2_latent import Text2Latent
from sft.models.latent_2_text import Latent2Text
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == 'cpu':
	print("WARNING: USING CPU")

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d", "--dataset", choices=["gsm8k", "4x4"], type=str, required=True
)
parser.add_argument(
    "-m", "--model", type=str, default="meta-llama/Llama-3.2-1B"
)
parser.add_argument("--checkpoints_dir", type=str, default="checkpoints/latent-cot-sft")
parser.add_argument(
    "--num_train", type=int, default=None, help="Number of training examples to use"
)
parser.add_argument(
	"-l", "--latent_pool", type=int, required=True
)

args = parser.parse_args()

checkpoints_path = os.path.join(
    args.checkpoints_dir, 
	args.dataset, 
	utils.get_cur_time_string()
)

model_id = args.model

tokenizer = AutoTokenizer.from_pretrained(model_id)

start_latent_string = "<|start-latent|>"
end_latent_string = "<|end-latent|>"
start_cot_string = "<|start-cot|>"
end_cot_string = "<|end-cot|>"

tokenizer.add_tokens(start_latent_string)
tokenizer.add_tokens(end_latent_string)
tokenizer.add_tokens(start_cot_string)
tokenizer.add_tokens(end_cot_string)

start_latent_id = tokenizer.convert_tokens_to_ids(start_latent_string)
end_latent_id = tokenizer.convert_tokens_to_ids(end_latent_string)
start_cot_id = tokenizer.convert_tokens_to_ids(start_cot_string)
end_cot_id = tokenizer.convert_tokens_to_ids(end_cot_string)

# model = Text2Latent(model_id, tokenizer)
model = Latent2Text(model_id, tokenizer)

# text_to_latent_ds = get_text_to_latent_dataset(
# 	tokenizer, 
# 	base_model.get_input_embeddings(), 
# 	start_latent_id, end_latent_id, 
# 	latent_pool=args.latent_pool, num_train=args.num_train
# )

latent_to_text_ds = get_latent_to_text_dataset(
	tokenizer,
	model.embedding(),
	start_latent_id, end_latent_id,
	start_cot_id, end_cot_id,
	latent_pool=args.latent_pool, num_train=args.num_train
)

optim = torch.optim.Adam(model.parameters(), lr=1e-5)

# text_to_latent_dataloader = DataLoader(text_to_latent_ds['train'], batch_size=32)
latent_to_text_dataloader = DataLoader(latent_to_text_ds['train'], collate_fn=collate_fn, batch_size=32)

# progress_bar = tqdm(range(len(text_to_latent_dataloader)))
progress_bar = tqdm(range(len(latent_to_text_dataloader)))

model = model.to(device)
# for batch_idx, batch in enumerate(text_to_latent_dataloader):
for batch_idx, batch in enumerate(latent_to_text_dataloader):
	batch = {k: v.to(device) for k, v in batch.items()}

	# loss = model(input_embeds=batch['input_embeds'], attention_mask=batch['attention_mask'], label_mask=batch['label_mask'])
	loss = model(input_embeds=batch['input_embeds'], attention_mask=batch['attention_mask'], labels=batch['labels'])

	progress_bar.set_description(f"Batch: {batch_idx}, Loss: {loss.item()}")

	optim.zero_grad()
	loss.backward()
	optim.step()

	progress_bar.update(1)

model.save_pretrained(checkpoints_path)
tokenizer.save_pretrained(checkpoints_path)
