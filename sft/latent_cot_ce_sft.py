'''
latent cot sft with cross entropy targets instead of l2
'''

import sys, os
sys.path.insert(0, os.path.abspath("."))  # hack for imports

import torch
from torch.utils.data import DataLoader
import argparse
from utils import utils
from data.dataset import get_latent_cot_ce_sft_dataset
from sft.models.latent_cot_model import LatentCOTModel
from sft.models.latent_tokenizer import LatentTokenizer
from tqdm.auto import tqdm
from data.multiplication_dataset import get_4x4_dataset, get_5x5_dataset
from data.gsm8k_dataset import get_gsm8k_dataset
import wandb

wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == 'cpu':
	print("WARNING: USING CPU")

parser = argparse.ArgumentParser()

parser.add_argument(
    "-c", "--config", type=str, required=True
)

config = utils.get_config(parser.parse_args().config)

run = wandb.init(
	project="Latent COT Cross-Entropy SFT",
	name=config.checkpoints_name,
	config=vars(config)
)

if config.no_cache:
	print("Disabling dataset caching")

	from datasets import disable_caching
	disable_caching()

base_checkpoints_path = os.path.join(
    config.checkpoints_dir, 
	"latent_cot_ce_sft",
	config.dataset, 
)

model_checkpoints_path = os.path.join(
	base_checkpoints_path,
	config.checkpoints_name,
	"model",	
)

tokenizer_checkpoints_path = os.path.join(
	base_checkpoints_path,
	config.checkpoints_name,
	"tokenizer",
)

print(f"Saving model @ {model_checkpoints_path}")
print(f"Saving tokenizer @ {tokenizer_checkpoints_path}")

tokenizer = LatentTokenizer(config.tokenizer)
tokenizer.save_pretrained(tokenizer_checkpoints_path)

if config.dataset == "4x4":
    base_ds = get_4x4_dataset(streaming=True)
elif config.dataset == "5x5":
	base_ds = get_5x5_dataset(streaming=True)
elif config.dataset == "gsm8k":
    base_ds = get_gsm8k_dataset(streaming=True)
else:
    raise ValueError(f"Unrecognized dataset: {config.dataset}")

def train_model(model: LatentCOTModel, dataset, checkpoints_path, lr):
	optim = torch.optim.AdamW(model.parameters(), lr=lr)

	# dataloader = DataLoader(dataset['train'], collate_fn=collate_fn, batch_size=config.batch_num)

	print("Don't forget to add collate_fn later")

	dataloader = DataLoader(dataset['train'], batch_size=config.batch_num)
	model = model.to(device)

	for epoch in range(config.epochs):
		progress_bar = tqdm(dataloader, desc=f"Epoch: {epoch}")

		for batch in progress_bar:
			batch = {k: v.to(device) for k, v in batch.items()}

			loss = model.ce_forward(**batch)

			optim.zero_grad()

			loss.backward()
			optim.step()

			loss_value = loss.item()

			postfix_dict = {}
			postfix_dict["loss"] = loss_value

			progress_bar.set_postfix(postfix_dict)
			run.log(postfix_dict)

		print(f"Finished Epoch ({epoch})")

		utils.torch_save(model, os.path.join(checkpoints_path, f"epoch_{epoch}", "model.pth"))

model = LatentCOTModel(config.model, tokenizer, freeze_embeddings=True)

ds = get_latent_cot_ce_sft_dataset(
	dataset=base_ds,
	tokenizer=tokenizer,
	embedding=model.latent_embedding,
	latent_pool=config.latent_pool,
	position_smoothing=config.position_smoothing,
) 

print("Training model")

utils.torch_save_sigint(model, os.path.join(model_checkpoints_path, "sigint", "model.pth"))

train_model(
	model=model,
	dataset=ds,
	checkpoints_path=model_checkpoints_path,
	lr=config.lr,
)

run.finish()

print(f"Model saved @ {model_checkpoints_path}")
print(f"Tokenizer saved @ {tokenizer_checkpoints_path}")
print("Finished training")