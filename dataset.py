from datasets import load_dataset
from trl import apply_chat_template

def get_gsm8k_dataset(tokenizer):
	def format_gsm8k_example(example):
		question = example["question"]
		answer = example["answer"]

		prompt = {"role": "user", "content": f"Question: {question} Let's think step by step. At the end, you MUST write the answer as an integer after '####'"}
		completion = {"role": "assistant", "content": f"Answer: {answer}"}

		return apply_chat_template({"prompt": [prompt], "completion": [completion]}, tokenizer)

	ds = load_dataset("openai/gsm8k", "main")

	ds['train'] = ds['train'].map(format_gsm8k_example)
	ds['test'] = ds['test'].map(format_gsm8k_example)

	return ds


def format_multiplication_example(example, tokenizer): 
	text = example['text']

	question = text.split('||')[0].strip()
	full_answer = text.split('||')[1]

	reasoning = full_answer.split("####")[0].strip()
	answer = full_answer.split("####")[1].strip()

	prompt = {"role": "user", "content": f"Question: {question}"}
	completion = {"role": "assistant", "content": f"Answer: {full_answer}"}

	return apply_chat_template({"prompt": [prompt], "completion": [completion]}, tokenizer)

def get_4x4_multiplication_dataset(tokenizer):
	ds = load_dataset("text", data_files={"train": "data/multiplication/4x4/train.txt", "test": "data/multiplication/4x4/valid.txt"})

	ds['train'] = ds['train'].map(format_multiplication_example, fn_kwargs={"tokenizer": tokenizer}, remove_columns=ds['train'].features)
	ds['test'] = ds['test'].map(format_multiplication_example, fn_kwargs={"tokenizer": tokenizer}, remove_columns=ds['test'].features)

	return ds
