"""

THIS FILE SHOULD BE RAN IN THE PARENT DIRECTORY, NOT INSIDE OF sft/

"""

import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM, apply_chat_template

import sys, os
sys.path.insert(0, os.path.abspath('.')) # hack for imports

from data import dataset

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", choices=['gsm8k', '4x4'], type=str, required=True)
parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument("--checkpoints_dir", type=str, default="checkpoints/cot-sft")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--num_train", type=int, default=None, help="Number of training examples to use")


args = parser.parse_args()

checkpoints_path = os.path.join(args.checkpoints_dir, args.dataset)
model_name = args.model

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load the dataset
ds = None

if args.dataset == "gsm8k":
    ds = dataset.get_gsm8k_dataset(tokenizer)
elif args.dataset == "4x4":
    ds = dataset.get_4x4_multiplication_dataset(tokenizer)

if args.num_train != None:
    ds['train'] = ds['train'].select(range(args.num_train))

print(
    f"Dataset loaded: {len(ds['train'])} training examples, {len(ds['test'])} test examples"
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(max_seq_length=2048,
    output_dir=checkpoints_path,
    report_to="none",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=8,
    optim="adamw_torch",
    learning_rate = 1e-4,
    logging_steps=10,
    weight_decay=0.01,
    warmup_steps=100,
    save_strategy="epoch"
)

trainer = SFTTrainer(
    model_name,
    train_dataset=ds['train'],
    args=training_args,
    peft_config=peft_config,
)

trainer.train()