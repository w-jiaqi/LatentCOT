import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import re
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

os.environ["HF_TOKEN"] = "your access token"

def load_model_and_tokenizer(model_path, is_adapter=False, base_model="meta-llama/Llama-3.2-1B-Instruct"):
    if is_adapter:
        base_tokenizer = AutoTokenizer.from_pretrained(base_model, token=os.environ["HF_TOKEN"])
        base_model_loaded = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            token=os.environ["HF_TOKEN"]
        )
        
        model = PeftModel.from_pretrained(
            base_model_loaded,
            model_path,
            device_map="auto"
        )
        
        return model, base_tokenizer
    else:
        print(f"Loading base model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=os.environ["HF_TOKEN"])
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            token=os.environ["HF_TOKEN"]
        )
        return model, tokenizer

def extract_answer_from_response(response):
    pattern = r"####\s*\[?\s*(\d+\.?\d*)\s*\]?"
    match = re.search(pattern, response)
    if match:
        answer_str = match.group(1)
        return float(answer_str) if '.' in answer_str else int(answer_str)
    pattern = r"####\s*(\d+\.?\d*)"
    match = re.search(pattern, response)
    if match:
        answer_str = match.group(1)
        return float(answer_str) if '.' in answer_str else int(answer_str)
    numbers = re.findall(r'\d+\.?\d*', response)
    if numbers:
        last_num = numbers[-1]
        return float(last_num) if '.' in last_num else int(last_num)
    
    return None

def extract_answer_from_dataset(answer_text):
    pattern = r"####\s*(\d+\.?\d*)"
    match = re.search(pattern, answer_text)
    if match:
        answer_str = match.group(1)
        return float(answer_str) if '.' in answer_str else int(answer_str)

    numbers = re.findall(r'\d+\.?\d*', answer_text)
    if numbers:
        last_num = numbers[-1]
        return float(last_num) if '.' in last_num else int(last_num)
    
    return None

def evaluate_model(model, tokenizer, dataset, num_samples=None):
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    correct = 0
    total = 0
    results = []
    
    for item in tqdm(dataset):
        question = item["question"]
        correct_answer = extract_answer_from_dataset(item["answer"])
        
        # prompt
        prompt = f"""Below is a grade school math problem. Solve it step-by-step and make sure to end your answer with the exact format: ####[final_answer]

For example, if the final answer is 5, your response should end with: ####[5]
If the final answer is 6.5, your response should end with: ####[6.5]

Question: {question}

Solution:"""
        
        # generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,  # greedy decoding, might want to change this, or not...
                do_sample=False
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        predicted_answer = extract_answer_from_response(response)
        is_correct = predicted_answer == correct_answer
        
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "question": question,
            "correct_answer": correct_answer,
            "model_response": response,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct
        })
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, results

def main():
    gsm8k = load_dataset("gsm8k", "main")
    test_dataset = gsm8k["test"]
    base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    fine_tuned_model_path = "models/llama-3.2-1B-gsm8k-sft-final/checkpoint-2805"
    num_samples = 100  # NONE for all
    
    results = {}

    # base model
    base_model, base_tokenizer = load_model_and_tokenizer(base_model_name, is_adapter=False)
    base_accuracy, base_detailed_results = evaluate_model(base_model, base_tokenizer, test_dataset, num_samples)
    
    results["base_model"] = {
        "accuracy": base_accuracy,
        "detailed_results": base_detailed_results
    }
    
    print(f"Base Model Accuracy: {base_accuracy:.4f}")
    
    # free up memory
    del base_model
    del base_tokenizer
    torch.cuda.empty_cache()
    
    # sft model
    ft_model, ft_tokenizer = load_model_and_tokenizer(
        fine_tuned_model_path, 
        is_adapter=True,
        base_model=base_model_name
    )
    ft_accuracy, ft_detailed_results = evaluate_model(ft_model, ft_tokenizer, test_dataset, num_samples)
    
    results["fine_tuned_model"] = {
        "accuracy": ft_accuracy,
        "detailed_results": ft_detailed_results
    }
    
    print(f"Fine-tuned Model Accuracy: {ft_accuracy:.4f}")

if __name__ == "__main__":
    main()