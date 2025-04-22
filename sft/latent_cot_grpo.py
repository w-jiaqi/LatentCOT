import sys, os
sys.path.insert(0, os.path.abspath("."))  # hack for imports

import torch
from torch.utils.data import DataLoader
import argparse
from utils import utils
from data.dataset import get_latent_cot_grpo_dataset, grpo_collate_fn
from sft.models.latent_cot_model import LatentCOTModel
from sft.models.latent_tokenizer import LatentTokenizer
from tqdm.auto import tqdm
from data.multiplication_dataset import get_4x4_dataset
from data.gsm8k_dataset import get_gsm8k_dataset
import wandb

wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == 'cpu':
	print("WARNING: USING CPU")

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d", "--dataset", choices=["gsm8k", "4x4"], type=str, required=True
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
parser.add_argument(
	"--checkpoints_dir", type=str, default="checkpoints/latent-cot-grpo"
)
parser.add_argument(
    "--num_train", type=int, default=None, help="Number of training examples to use"
)
parser.add_argument(
	"--batch_num", type=int, default=8
)
parser.add_argument(
	"--max_new_latents", type=int
)
parser.add_argument(
	"--checkpoints_name", type=str, default=utils.get_cur_time_string(), help="Name of checkpoints folder underneath checkpoints_dir"
)

args = parser.parse_args()

run = wandb.init(
	project="Latent COT GRPO (not really)",
	config=vars(args)
)

base_checkpoints_path = os.path.join(
    args.checkpoints_dir, 
	args.dataset, 
)

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

print(f"Saving model @ {model_checkpoints_path}")
print(f"Saving tokenizer @ {tokenizer_checkpoints_path}")

model_id = args.model

tokenizer = LatentTokenizer(args.tokenizer)

if args.dataset == "4x4":
    base_ds = get_4x4_dataset(streaming=False)
elif args.dataset == "gsm8k":
    base_ds = get_gsm8k_dataset(streaming=False)
else:
    raise ValueError(f"Unrecognized dataset: {args.dataset}")

if base_ds is None:
	print("No dataset found, exiting")
	sys.exit()

ds = get_latent_cot_grpo_dataset(
	dataset=base_ds,
	tokenizer=tokenizer,
)

def train_model(model: LatentCOTModel, dataset, checkpoints_path):
	token_optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

	dataloader = DataLoader(dataset['train'], batch_size=args.batch_num, collate_fn=grpo_collate_fn)
	model = model.to(device)

	for epoch in range(args.epochs):
		progress_bar = tqdm(dataloader, desc=f"Epoch: {epoch}")

		for batch in progress_bar:
			batch = {k: v.to(device) for k, v in batch.items()}

			loss = model.grpo_forward(
				**batch,
				max_new_latents=args.max_new_latents,
			)

			progress_bar.set_postfix({'loss': loss.item()})
			run.log({'loss': loss.item()})
			
			token_optimizer.zero_grad()
			loss.backward()

			token_optimizer.step()

		print(f"Finished Epoch ({epoch})")

		model.save_pretrained(os.path.join(checkpoints_path, f"epoch_{epoch}"))
		tokenizer.save_pretrained(os.path.join(checkpoints_path, f"epoch_{epoch}"))

	model.save_pretrained(checkpoints_path)


model = LatentCOTModel(model_id, tokenizer)

print("Training model")

import signal

def handle_sig(sig, frame):
	print(f"Saving model @ {model_checkpoints_path}")
	model.save_pretrained(model_checkpoints_path)

	print(f"Saving tokenizer @ {tokenizer_checkpoints_path}")
	tokenizer.save_pretrained(tokenizer_checkpoints_path)

	sys.exit(0)

signal.signal(signal.SIGINT, handle_sig) # save model on ctrl-c

train_model(
	model=model,
	dataset=ds,
	checkpoints_path=model_checkpoints_path,
)

tokenizer.save_pretrained(tokenizer_checkpoints_path)

run.finish()

print(f"Model saved @ {model_checkpoints_path}")
print(f"Tokenizer saved @ {tokenizer_checkpoints_path}")
print("Finished training")