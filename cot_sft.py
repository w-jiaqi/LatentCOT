import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

sft_path = "./models/cot-sft/llama-3.2-1B-gsm8k-sft-final"
model_name = "meta-llama/Llama-3.2-1B-Instruct"

# load the dataset
ds = load_dataset("openai/gsm8k", "main")
print(
    f"Dataset loaded: {len(ds['train'])} training examples, {len(ds['test'])} test examples"
)


def format_gsm8k_example(example):
    """Format GSM8K examples into prompt-completion pairs"""
    question = example["question"]
    full_answer = example["answer"]

    answer_parts = full_answer.split("####")
    reasoning = answer_parts[0].strip()
    final_answer = answer_parts[1].strip() if len(answer_parts) > 1 else ""

    prompt = f"""Solve the following math problem step by step:

        {question}

        Think through this problem step by step:"""

    completion = f"""{reasoning}

        Therefore, the answer is {final_answer}"""

    return {"prompt": prompt, "completion": completion}


# format the dataset
formatted_train_ds = ds["train"].map(format_gsm8k_example)
formatted_test_ds = ds["test"].map(format_gsm8k_example)

model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.add_special_tokens({"pad_token":"<|pad|>"})
model.resize_token_embeddings(len(tokenizer))
pad_id = tokenizer.convert_tokens_to_ids("<|pad|>")

model.config.pad_token_id=pad_id

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(max_seq_length=2048, output_dir=sft_path, report_to="none", num_train_epochs=3, per_device_train_batch_size=8, gradient_accumulation_steps=2, gradient_checkpointing=True, optim="adamw_torch", learning_rate = 1e-4, logging_steps=10, weight_decay=0.01, warmup_steps=100, save_strategy="epoch")

trainer = SFTTrainer(
    model_name,
    train_dataset=formatted_train_ds,
    args=training_args,
    peft_config=peft_config,
)

trainer.train()

trainer.save_model()
