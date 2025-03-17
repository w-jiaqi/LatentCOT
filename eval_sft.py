import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset

model_name = "meta-llama/Llama-3.2-1B-Instruct"

print("Loading Model")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading dataset")
ds = load_dataset("openai/gsm8k", "main")

pipe = pipeline(
    "text-generation", 
    model=model_name, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

for out in pipe(ds):
	print(out)