from datasets import load_dataset
from transformers import AutoTokenizer

# 1) Load GSM8K from Hugging Face
dataset = load_dataset("gsm8k", "main")  # has "train" and "test" splits

# 2) Pick a tokenizer (change to your model of interest)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

max_question_len = 0
max_reasoning_len = 0

# 3) Loop through the training set
for example in dataset["train"]:
    question_text = example["question"]
    answer_text = example["answer"]

    # If there's a "####", treat everything before it as chain-of-thought "reasoning".
    # Otherwise, we consider the entire answer as "reasoning".
    parts = answer_text.split("####")
    if len(parts) > 1:
        reasoning_text = parts[0].strip()  # chain-of-thought
    else:
        reasoning_text = answer_text.strip()

    # Tokenize question and reasoning
    q_tokens = tokenizer(question_text)["input_ids"]
    r_tokens = tokenizer(reasoning_text)["input_ids"]

    # Track max lengths
    if len(q_tokens) > max_question_len:
        max_question_len = len(q_tokens)
    if len(r_tokens) > max_reasoning_len:
        max_reasoning_len = len(r_tokens)

print("Max token length for questions (train):", max_question_len)
print("Max token length for reasonings (train):", max_reasoning_len)

# Optional: Repeat for the "test" split if desired
max_question_len_test = 0
max_reasoning_len_test = 0
for example in dataset["test"]:
    question_text = example["question"]
    answer_text = example["answer"]
    parts = answer_text.split("####")
    if len(parts) > 1:
        reasoning_text = parts[0].strip()
    else:
        reasoning_text = answer_text.strip()

    q_tokens = tokenizer(question_text)["input_ids"]
    r_tokens = tokenizer(reasoning_text)["input_ids"]

    if len(q_tokens) > max_question_len_test:
        max_question_len_test = len(q_tokens)
    if len(r_tokens) > max_reasoning_len_test:
        max_reasoning_len_test = len(r_tokens)

print("Max token length for questions (test):", max_question_len_test)
print("Max token length for reasonings (test):", max_reasoning_len_test)
