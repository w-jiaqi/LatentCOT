import sys, os
sys.path.insert(0, os.path.abspath("."))  # hack for imports

import torch
from torch.utils.data import DataLoader
import argparse
from utils import utils
from data.dataset import get_latent_cot_sft_dataset, collate_fn
from sft.models.text_2_latent import Text2Latent
from sft.models.latent_2_text import Latent2Text
from sft.models.latent_cot_model import LatentCOTModel
from sft.models.latent_tokenizer import LatentTokenizer
from tqdm.auto import tqdm
from data.multiplication_dataset import get_4x4_dataset
from data.gsm8k_dataset import get_gsm8k_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == 'cpu':
	print("WARNING: USING CPU")

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d", "--dataset", choices=["gsm8k", "4x4"], type=str, required=True
)
parser.add_argument(
	"-l", "--latent_pool", type=int, required=True, help="Number of embeddings to mean pool for sft"
)
parser.add_argument(
	"-e", "--epochs", type=int, default=1
)
parser.add_argument(
    "-m", "--model", type=str, default="meta-llama/Llama-3.2-1B"
)
parser.add_argument(
    "-t", "--tokenizer", type=str, default="meta-llama/Llama-3.2-1B"
)
parser.add_argument("--checkpoints_dir", type=str, default="checkpoints/latent-cot-sft")
parser.add_argument(
    "--num_train", type=int, default=None, help="Number of training examples to use"
)
parser.add_argument(
	"--batch_num", type=int, default=32
)
parser.add_argument(
	"--num_proc", type=int, default=None
)
parser.add_argument(
	"--checkpoints_name", type=str, default=utils.get_cur_time_string(), help="Name of checkpoints folder underneath checkpoints_dir"
)
parser.add_argument(
	"--no_cache", action='store_true', help="Disable caching for datasets (helps with disk space)"
)
# parser.add_argument(
# 	"--skip_t2l", action="store_true", help="Skip text to latent training"
# )

# parser.add_argument(
# 	"--skip_l2t", action="store_true", help="Skip latent to text training"
# )

# parser.add_argument(
# 	"--t2l_lr", type=float, default=1e-2, help="Learning rate for text to latents"
# )

# parser.add_argument(
# 	"--l2t_lr", type=float, default=1e-5, help="Learning rate for latents to text"
# )


args = parser.parse_args()

if args.no_cache:
	print("Disabling dataset caching")

	from datasets import disable_caching
	disable_caching()

base_checkpoints_path = os.path.join(
    args.checkpoints_dir, 
	args.dataset, 
)

# text_to_latent_checkpoints_path = os.path.join(
# 	base_checkpoints_path,
# 	args.checkpoints_name,
# 	"text_to_latent",
# )

# latent_to_text_checkpoints_path = os.path.join(
# 	base_checkpoints_path,
# 	args.checkpoints_name,
# 	"latent_to_text",
# )

model_checkpoints_path = os.path.join(
	base_checkpoints_path,
	args.checkpoints_name,
	"model",	
)

tokenizer_checkpoints_path = os.path.join(
	base_checkpoints_path,
	args.checkpoints_name,
	"tokenizer",
)

# print(f"Saving text2latent @ {text_to_latent_checkpoints_path}")
# print(f"Saving latent2text @ {latent_to_text_checkpoints_path}")

print(f"Saving model @ {model_checkpoints_path}")
print(f"Saving tokenizer @ {tokenizer_checkpoints_path}")

model_id = args.model

tokenizer = LatentTokenizer(args.tokenizer)

base_ds = None

if args.dataset == "4x4":
    base_ds = get_4x4_dataset(num_train=args.num_train, num_proc=args.num_proc)
elif args.dataset == "gsm8k":
    base_ds = get_gsm8k_dataset(num_train=args.num_train, num_proc=args.num_proc)
else:
    raise ValueError(f"Unrecognized dataset: {args.dataset}")

if base_ds is None:
	print("No dataset found, exiting")
	sys.exit()

def train_model(model, dataset, checkpoints_path, learning_rate):
	optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
	dataloader = DataLoader(dataset['train'], collate_fn=collate_fn, batch_size=args.batch_num)
	model = model.to(device)

	for epoch in range(args.epochs):
		progress_bar = tqdm(dataloader, desc=f"Epoch: {epoch}")

		for batch in progress_bar:
			batch = {k: v.to(device) for k, v in batch.items()}

			loss = model(**batch)

			token_loss = loss.token_loss
			latents_loss = loss.latents_loss
			total_loss = token_loss + latents_loss

			progress_bar.set_postfix(token_loss=token_loss.item(), latents_loss=latents_loss.item())

			optim.zero_grad()
			total_loss.backward()
			optim.step()

		print(f"Finished Epoch ({epoch})")

	model.save_pretrained(checkpoints_path)

# def train_text_to_latent():
# 	model = Text2Latent(model_id, tokenizer)

# 	ds = get_text_to_latent_dataset(
# 		dataset=base_ds,
# 		tokenizer=tokenizer, 
# 		embedding=model.embedding, 
# 		latent_pool=args.latent_pool, 
# 	)

# 	print("Training text2latent")

# 	train_model(
# 		model,
# 		ds,
# 		text_to_latent_checkpoints_path,
# 		args.t2l_lr
# 	)

# def train_latent_to_text():
# 	model = Latent2Text(model_id, tokenizer)

# 	ds = get_latent_to_text_dataset(
# 		dataset=base_ds,
# 		tokenizer=tokenizer, 
# 		embedding=model.embedding, 
# 		latent_pool=args.latent_pool, 
# 	)

# 	print("Training latent2text")

# 	train_model(
# 		model,
# 		ds,
# 		latent_to_text_checkpoints_path,
# 		args.l2t_lr
# 	)

# if not args.skip_t2l:
# 	train_text_to_latent()

# if not args.skip_l2t:
# 	train_latent_to_text()

model = LatentCOTModel(model_id, tokenizer)

ds = get_latent_cot_sft_dataset(
	dataset=base_ds,
	tokenizer=tokenizer,
	embedding=model.embedding,
	latent_pool=args.latent_pool,
)

print("Training model")

train_model(
	model,
	ds,
	model_checkpoints_path,
	1e-5
)

tokenizer.save_pretrained(tokenizer_checkpoints_path)