"""

THIS FILE SHOULD BE RAN IN THE PARENT DIRECTORY, NOT INSIDE OF sft/

"""

import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainingArguments,
)

import sys, os

sys.path.insert(0, os.path.abspath("."))  # hack for imports

from data import dataset, multiplication_dataset
import utils.utils as utils

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d", "--dataset", choices=["gsm8k", "4x4", "5x5"], type=str, required=True
)
parser.add_argument(
    "-m", "--model", type=str, default="meta-llama/Llama-3.2-1B"
)
parser.add_argument("--checkpoints_dir", type=str, default="checkpoints/cot-sft")
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument(
    "--num_train", type=int, default=None, help="Number of training examples to use"
)
parser.add_argument(
    "--checkpoints_name", type=str, default=utils.get_cur_time_string()
)

args = parser.parse_args()

checkpoints_path = os.path.join(
    args.checkpoints_dir, 
    args.dataset, 
    args.checkpoints_name,
)

model_name = args.model

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load the dataset
base_ds = None

print(vars(args))

if args.dataset == "gsm8k":
    base_ds = dataset.get_gsm8k_dataset(tokenizer)
elif args.dataset == "4x4":
    base_ds = multiplication_dataset.get_4x4_dataset(num_train=args.num_train)
elif args.dataset == "5x5":
    base_ds = multiplication_dataset.get_5x5_dataset(num_train=args.num_train)

ds = dataset.get_cot_sft_dataset(base_ds, tokenizer)

print(
    f"Dataset loaded: {len(ds['train'])} training examples, {len(ds['test'])} test examples"
)

training_args = TrainingArguments(
    output_dir=checkpoints_path,
    report_to="wandb",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=8,
    optim="adamw_torch",
    learning_rate=1e-4,
    logging_steps=10,
    weight_decay=0.01,
    warmup_steps=100,
    save_strategy="steps",
)

trainer = Trainer(
    model,
    train_dataset=ds["train"],
    args=training_args,
)

trainer.train()
