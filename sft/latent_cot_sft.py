import sys, os
sys.path.insert(0, os.path.abspath("."))  # hack for imports

import torch
from torch.utils.data import DataLoader
import argparse
from utils import utils
from data.dataset import get_text_to_latent_dataset, get_latent_to_text_dataset, collate_fn
from sft.models.text_2_latent import Text2Latent
from sft.models.latent_2_text import Latent2Text
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
	"-l", "--latent_pool", type=int, required=True, help="Number of embeddings to mean pool for sft"
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

args = parser.parse_args()

if args.no_cache:
	print("Disabling dataset caching")

	from datasets import disable_caching
	disable_caching()

base_checkpoints_path = os.path.join(
    args.checkpoints_dir, 
	args.dataset, 
)

text_to_latent_checkpoints_path = os.path.join(
	base_checkpoints_path,
	args.checkpoints_name,
	"text_to_latent",
)

latent_to_text_checkpoints_path = os.path.join(
	base_checkpoints_path,
	args.checkpoints_name,
	"latent_to_text",
)

tokenizer_checkpoints_path = os.path.join(
	base_checkpoints_path,
	args.checkpoints_name,
	"tokenizer",
)

print(f"Saving text2latent @ {text_to_latent_checkpoints_path}")
print(f"Saving latent2text @ {latent_to_text_checkpoints_path}")

model_id = args.model

tokenizer = LatentTokenizer(args.tokenizer)

# text_to_latent_model = Text2Latent(model_id, tokenizer)
latent_to_text_model = Latent2Text(model_id, tokenizer)

if args.dataset == "4x4":
    base_ds = get_4x4_dataset(num_train=args.num_train, num_proc=args.num_proc)
elif args.dataset == "gsm8k":
    base_ds = get_gsm8k_dataset(num_train=args.num_train, num_proc=args.num_proc)
else:
    raise ValueError(f"Unrecognized dataset: {args.dataset}")

if base_ds is None:
	print("No dataset found, exiting")

# text_to_latent_ds = get_text_to_latent_dataset(
# 	dataset=base_ds,
# 	tokenizer=tokenizer, 
# 	embedding=text_to_latent_model.embedding, 
# 	latent_pool=args.latent_pool, 
# )

latent_to_text_ds = get_latent_to_text_dataset(
	dataset=base_ds,
	tokenizer=tokenizer, 
	embedding=latent_to_text_model.embedding, 
	latent_pool=args.latent_pool, 
)

def train_model(model, dataset, checkpoints_path):
	optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
	dataloader = DataLoader(dataset['train'], collate_fn=collate_fn, batch_size=args.batch_num)

	model = model.to(device)

	progress_bar = tqdm(range(len(dataloader)))

	for batch_idx, batch in enumerate(dataloader):
		batch = {k: v.to(device) for k, v in batch.items()}

		loss = model(**batch)

		progress_bar.set_description(f"Batch: {batch_idx}, Loss: {loss.item()}")

		optim.zero_grad()
		loss.backward()
		optim.step()

		progress_bar.update(1)

	model.save_pretrained(checkpoints_path)

# print("Training text2latent")
# train_model(
# 	text_to_latent_model,
# 	text_to_latent_ds,
# 	text_to_latent_checkpoints_path
# )
print("Training latent2text")
train_model(
	latent_to_text_model,
	latent_to_text_ds,
	latent_to_text_checkpoints_path
)

tokenizer.save_pretrained(tokenizer_checkpoints_path)