import sys, os
sys.path.insert(0, os.path.abspath("."))  # hack for imports

import torch
from torch.utils.data import DataLoader
import argparse
from utils import utils
from data.dataset import get_latent_cot_sft_dataset, collate_fn
from sft.models.text_2_latent import Text2Latent
from sft.models.latent_2_text import Latent2Text
from sft.models.latent_cot_model import LatentCOTModel, LossType
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
parser.add_argument(
	"--checkpoints_dir", type=str, default="checkpoints/latent-cot-sft"
)
parser.add_argument(
    "--num_train", type=int, default=None, help="Number of training examples to use"
)
parser.add_argument(
	"--batch_num", type=int, default=32
)
parser.add_argument(
	"--checkpoints_name", type=str, default=utils.get_cur_time_string(), help="Name of checkpoints folder underneath checkpoints_dir"
)
parser.add_argument(
	"--latents_lr", type=float, default=1e-5, help="Latents learning rate"
)
parser.add_argument(
	"--token_lr", type=float, default=1e-5, help="Token learning rate"
)
parser.add_argument(
	"--no_cache", action='store_true', help="Disable caching for datasets (helps with disk space)"
)
parser.add_argument(
	"--skip_latents", action='store_true', help="Skip training of latent loss"
)
parser.add_argument(
	"--skip_tokens", action='store_true', help="Skip training of token loss"
)
parser.add_argument(
	"--tie_weights", action='store_true', help="Tie weights of input and output embeddings"
)

args = parser.parse_args()

run = wandb.init(
	project="Latent COT SFT",
	config=vars(args)
)

if args.no_cache:
	print("Disabling dataset caching")

	from datasets import disable_caching
	disable_caching()

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

base_ds = None

if args.dataset == "4x4":
    base_ds = get_4x4_dataset(num_train=args.num_train)
elif args.dataset == "gsm8k":
    base_ds = get_gsm8k_dataset(num_train=args.num_train)
else:
    raise ValueError(f"Unrecognized dataset: {args.dataset}")

if base_ds is None:
	print("No dataset found, exiting")
	sys.exit()

def train_model(model: LatentCOTModel, dataset, checkpoints_path, latents_lr, token_lr):

	latents_optimizer = torch.optim.Adam(model.parameters(), lr=latents_lr)
	token_optimizer = torch.optim.Adam(model.parameters(), lr=token_lr)

	dataloader = DataLoader(dataset['train'], collate_fn=collate_fn, batch_size=args.batch_num)
	model = model.to(device)

	for epoch in range(args.epochs):
		progress_bar = tqdm(dataloader, desc=f"Epoch: {epoch}")

		for batch in progress_bar:
			batch = {k: v.to(device) for k, v in batch.items()}

			latents_loss_value = None
			token_loss_value = None

			if not args.skip_latents:
				latents_loss = model(**batch, output_loss = LossType.LATENTS)

				latents_optimizer.zero_grad()

				latents_loss.backward(retain_graph=True)
				latents_optimizer.step()

				latents_loss_value = latents_loss.item()

			if not args.skip_tokens:
				token_loss = model(**batch, output_loss = LossType.TOKEN)

				token_optimizer.zero_grad()

				token_loss.backward()
				token_optimizer.step()

				token_loss_value = token_loss.item()

			postfix_dict = {}

			if not latents_loss_value == None:
				postfix_dict['latents_loss'] = latents_loss_value
			if not token_loss_value == None:
				postfix_dict['token_loss'] = token_loss_value

			progress_bar.set_postfix(**postfix_dict)
			run.log(postfix_dict)

		print(f"Finished Epoch ({epoch})")

	model.save_pretrained(checkpoints_path)


model = LatentCOTModel(model_id, tokenizer, tie_weights=args.tie_weights)

ds = get_latent_cot_sft_dataset(
	dataset=base_ds,
	tokenizer=tokenizer,
	embedding=model.embedding,
	latent_pool=args.latent_pool,
) 

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
	latents_lr=args.latents_lr,
	token_lr=args.token_lr
)

tokenizer.save_pretrained(tokenizer_checkpoints_path)

run.finish()