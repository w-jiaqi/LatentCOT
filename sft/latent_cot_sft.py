import sys, os
sys.path.insert(0, os.path.abspath("."))  # hack for imports

import torch
from torch.utils.data import DataLoader
import argparse
from utils import utils
from data.dataset import get_latent_cot_sft_dataset, collate_fn
from sft.models.latent_cot_model import LatentCOTModel, LossType
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
	project="Latent COT SFT",
	name=config.checkpoints_name,
	config=vars(config)
)

if config.no_cache:
	print("Disabling dataset caching")

	from datasets import disable_caching
	disable_caching()

base_checkpoints_path = os.path.join(
    config.checkpoints_dir, 
	"latent_cot_sft",
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

def train_model(model: LatentCOTModel, dataset, checkpoints_path, latents_lr, token_lr):

	latents_optimizer = torch.optim.Adam(model.parameters(), lr=latents_lr)
	token_optimizer = torch.optim.Adam(model.parameters(), lr=token_lr)

	dataloader = DataLoader(dataset['train'], collate_fn=collate_fn, batch_size=config.batch_num)
	model = model.to(device)

	for epoch in range(config.epochs):
		progress_bar = tqdm(dataloader, desc=f"Epoch: {epoch}")

		for batch in progress_bar:
			batch = {k: v.to(device) for k, v in batch.items()}

			latents_loss_value = None
			token_loss_value = None

			if not config.skip_latents:
				latents_loss = model(**batch, output_loss = LossType.LATENTS)

				latents_optimizer.zero_grad()

				latents_loss.backward(retain_graph=True)
				latents_optimizer.step()

				latents_loss_value = latents_loss.item()

			if not config.skip_tokens:
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

		utils.torch_save(model, os.path.join(checkpoints_path, f"epoch_{epoch}", "model.pth"))

model = LatentCOTModel(config.model, tokenizer, freeze_embeddings=True)

ds = get_latent_cot_sft_dataset(
	dataset=base_ds,
	tokenizer=tokenizer,
	embedding=model.embedding,
	latent_pool=config.latent_pool,
) 

print("Training model")

utils.torch_save_sigint(model, os.path.join(model_checkpoints_path, "sigint", "model.pth"))

train_model(
	model=model,
	dataset=ds,
	checkpoints_path=model_checkpoints_path,
	latents_lr=config.latents_lr,
	token_lr=config.token_lr
)

run.finish()

print(f"Model saved @ {model_checkpoints_path}")
print(f"Tokenizer saved @ {tokenizer_checkpoints_path}")
print("Finished training")