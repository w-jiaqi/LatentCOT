import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_answer(question, model, tokenizer, max_new_tokens=512, temperature=0.1, top_p=0.95):
    prompt = f"""Solve the following math problem step by step:

{question}

Think through this problem step by step:"""
    # tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # generate the answer
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True
    )
    # decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

def main():
    model_path = "./models/llama-3.2-1B-gsm8k-sft-final"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    test_question = "If a train travels 100 miles in 2 hours, what is its speed in mph?"
    print("Test Question:")
    print(test_question)
    
    print("\nGenerated Answer:")
    answer = generate_answer(test_question, model, tokenizer)
    print(answer)

if __name__ == "__main__":
    main()