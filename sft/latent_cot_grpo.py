import sys, os
sys.path.insert(0, os.path.abspath("."))  # hack for imports

import torch
from torch.utils.data import DataLoader
import argparse
from data.dataset import get_latent_cot_grpo_dataset, grpo_collate_fn
from sft.models.latent_cot_model import LatentCOTModel
from sft.models.latent_tokenizer import LatentTokenizer
from tqdm.auto import tqdm
from data.multiplication_dataset import get_4x4_dataset, get_5x5_dataset
from data.gsm8k_dataset import get_gsm8k_dataset
import wandb
from utils.utils import torch_save, torch_save_sigint, get_config

wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == 'cpu':
        print("WARNING: USING CPU")

parser = argparse.ArgumentParser()

parser.add_argument(
    "-c", "--config", type=str, required=True
)

config = get_config(parser.parse_args().config)

run = wandb.init(
        project="Latent COT GRPO (not really)",
        name=config.checkpoints_name,
        config=vars(config)
)

base_checkpoints_path = os.path.join(
        config.checkpoints_dir, 
        "latent_cot_grpo",
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
    base_ds = get_4x4_dataset(streaming=False)
elif config.dataset == "5x5":
    base_ds = get_5x5_dataset(streaming=False)
elif config.dataset == "gsm8k":
    base_ds = get_gsm8k_dataset(streaming=False)
else:
    raise ValueError(f"Unrecognized dataset: {config.dataset}")

ds = get_latent_cot_grpo_dataset(
        dataset=base_ds,
        tokenizer=tokenizer,
)

def train_model(model: LatentCOTModel, dataset, checkpoints_path):
        token_optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

        dataloader = DataLoader(dataset['train'], batch_size=config.batch_num, collate_fn=grpo_collate_fn)

        for epoch in range(config.epochs):
                progress_bar = tqdm(dataloader, desc=f"Epoch: {epoch}")

                for idx, batch in enumerate(progress_bar):
                        batch = {k: v.to(device) for k, v in batch.items()}

                        loss = model.grpo_forward(
                                **batch,
                                max_new_latents=config.max_new_latents,
                                unembed_latents=config.unembed_latents,
                                dynamically_stop=config.dynamically_stop,
                        )

                        progress_bar.set_postfix({'loss': loss.item()})
                        run.log({'loss': loss.item()})
                        
                        token_optimizer.zero_grad()
                        loss.backward()

                        token_optimizer.step()

                        if idx != 0 and idx % config.save_steps == 0: 
                            torch_save(model, os.path.join(checkpoints_path, f"epoch_{epoch}_{idx}", "model.pth"))

                print(f"Finished Epoch ({epoch})")

                torch_save(model, os.path.join(checkpoints_path, f"epoch_{epoch}", "model.pth"))

        print("Finished training")

model = LatentCOTModel(config.model, tokenizer, freeze_embeddings=config.freeze_embeddings).to(device)

print("Training model")

torch_save_sigint(model, os.path.join(model_checkpoints_path, "sigint", "model.pth"))

train_model(
        model=model,
        dataset=ds,
        checkpoints_path=model_checkpoints_path,
)

run.finish()

print(f"Model saved @ {model_checkpoints_path}")
print(f"Tokenizer saved @ {tokenizer_checkpoints_path}")
print("Finished training")
