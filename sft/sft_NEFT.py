import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# load the dataset
ds = load_dataset("openai/gsm8k", "main")
print(f"Dataset loaded: {len(ds['train'])} training examples, {len(ds['test'])} test examples")

model_path = "./models/llama-3.2-1B-gsm8k-sft-final"

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
    
    return {
        "prompt": prompt,
        "completion": completion
    }

# format the dataset
formatted_train_ds = ds["train"].map(format_gsm8k_example)
formatted_test_ds = ds["test"].map(format_gsm8k_example)

# print a sample formatted example
print("\nSample formatted example:")
sample_idx = 0
print(f"PROMPT:\n{formatted_train_ds[sample_idx]['prompt']}")
print(f"\nCOMPLETION:\n{formatted_train_ds[sample_idx]['completion']}")

# load the model
model_name = "meta-llama/Llama-3.2-1B-Instruct"

# configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# configure lora
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=model_path,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="adamw_torch",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    report_to="tensorboard",
    warmup_steps=100,
)

response_template = " " 

# data collator
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
    mlm=False
)

# run the trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_train_ds,
    eval_dataset=formatted_test_ds,
    data_collator=collator,
    tokenizer=tokenizer,
    max_seq_length=1024,
    neftune_noise_alpha=5
)

trainer.train()

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def generate_answer(question, model, tokenizer):
    prompt = f"""Solve the following math problem step by step:

{question}

Think through this problem step by step:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.95,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

if __name__ == "__main__":
    sample_test_question = ds["test"][0]["question"]
    generated_answer = generate_answer(sample_test_question, model, tokenizer)
    
    print(f"\nSample Question: {sample_test_question}")
    print(f"\nGenerated Answer:\n{generated_answer}")
    print(f"\nGround Truth:\n{ds['test'][0]['answer']}")