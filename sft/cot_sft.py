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

from data import dataset, gsm8k_dataset, multiplication_dataset
import utils.utils as utils

parser = argparse.ArgumentParser()

parser.add_argument(
    "-c", "--config", type=str, required=True
)

config = utils.get_config(parser.parse_args().config)

checkpoints_path = os.path.join(
    config.checkpoints_dir, 
    "cot_sft",
    config.dataset, 
    config.checkpoints_name,
)

model_name = config.model

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

if config.dataset == "gsm8k":
    base_ds = gsm8k_dataset.get_gsm8k_dataset(streaming=False)
elif config.dataset == "4x4":
    base_ds = multiplication_dataset.get_4x4_dataset(streaming=False)
elif config.dataset == "5x5":
    base_ds = multiplication_dataset.get_5x5_dataset(streaming=False)

ds = dataset.get_cot_sft_dataset(base_ds, tokenizer, skip_cot=config.skip_cot)

print(
    f"Dataset loaded: {len(ds['train'])} training examples, {len(ds['valid'])} validation examples"
)

num_train_epochs = getattr(config, "epochs", 3)
per_device_train_batch_size = getattr(config, "per_device_train_batch_size", 32)
learning_rate = getattr(config, "learning_rate", 1e-4)
logging_steps = getattr(config, "logging_steps", 10)
weight_decay = getattr(config, "weight_decay", 0.01)
warmup_steps = getattr(config, "warmup_steps", 100)
save_steps = getattr(config, "save_steps", 5000)

os.environ["WANDB_PROJECT"] = "Base SFT"

training_args = TrainingArguments(
    output_dir=checkpoints_path,
    report_to="wandb",
    run_name=config.checkpoints_name,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    optim="adamw_torch",
    learning_rate=learning_rate,
    logging_steps=logging_steps,
    weight_decay=weight_decay,
    warmup_steps=warmup_steps,
    save_strategy="steps",
    save_steps=save_steps,
)

trainer = Trainer(
    model,
    train_dataset=ds["train"],
    args=training_args,
)

trainer.train()
