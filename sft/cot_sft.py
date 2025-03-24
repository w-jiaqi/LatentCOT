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
    "-d", "--dataset", choices=["gsm8k", "4x4"], type=str, required=True
)
parser.add_argument(
    "-m", "--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct"
)
parser.add_argument("--checkpoints_dir", type=str, default="checkpoints/cot-sft")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument(
    "--num_train", type=int, default=None, help="Number of training examples to use"
)

args = parser.parse_args()

checkpoints_path = os.path.join(
    args.checkpoints_dir, 
    args.dataset, 
    utils.get_cur_time_string()
)

model_name = args.model

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load the dataset
ds = None

print(vars(args))

if args.dataset == "gsm8k":
    ds = dataset.get_gsm8k_dataset(tokenizer)
elif args.dataset == "4x4":
    ds = multiplication_dataset.get_4x4_multiplication_dataset(tokenizer, num_train=args.num_train)

print(
    f"Dataset loaded: {len(ds['train'])} training examples, {len(ds['test'])} test examples"
)

example_train = ds["train"][0]

print(example_train)

training_args = TrainingArguments(
    output_dir=checkpoints_path,
    report_to="none",
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
