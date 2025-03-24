from datasets import load_dataset
from trl import apply_chat_template


def get_gsm8k_dataset(tokenizer):
    def format_gsm8k_example(example):
        question = example["question"]
        answer = example["answer"]

        prompt = {
            "role": "user",
            "content": f"Question: {question} Let's think step by step. At the end, you MUST write the answer as an integer after '####'",
        }
        completion = {"role": "assistant", "content": f"Answer: {answer}"}

        return apply_chat_template(
            {"prompt": [prompt], "completion": [completion]}, tokenizer
        )

    ds = load_dataset("openai/gsm8k", "main")

    ds["train"] = ds["train"].map(format_gsm8k_example)
    ds["test"] = ds["test"].map(format_gsm8k_example)

    return ds