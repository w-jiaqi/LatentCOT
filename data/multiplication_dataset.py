import sys, os
sys.path.insert(0, os.path.abspath("."))  # hack for imports

from datasets import load_dataset, DatasetDict, Dataset
from typing import Union, Optional

TRAIN_4x4_PATH = "data/multiplication/4x4/train.txt"
VALID_4x4_PATH = "data/multiplication/4x4/valid.txt"

TRAIN_5x5_PATH = "data/multiplication/5x5/train.txt"
VALID_5x5_PATH = "data/multiplication/5x5/valid.txt"

def preprocess_fn(example):
    text = example['text']

    question = text.split("||")[0]
    full_answer = text.split("||")[1]
    reasoning = full_answer.split("####")[0].strip()
    answer = full_answer.split("####")[1].strip()

    return {
        "question": question,
        "reasoning": reasoning,
        "answer": answer,
    }

def get_4x4_dataset(streaming: bool) -> Union[DatasetDict, Dataset]:
    ds = load_dataset(
        "text",
        data_files={
            "train": TRAIN_4x4_PATH,
            "valid": VALID_4x4_PATH,
        },
        streaming=streaming
    )

    ds = ds.map(preprocess_fn, batched=False, remove_columns=['text'])
    
    return ds

def get_5x5_dataset(streaming: bool) -> Union[DatasetDict, Dataset]:
    ds = load_dataset(
        "text",
        data_files={
            "train": TRAIN_5x5_PATH,
            "valid": VALID_5x5_PATH,
        },
        streaming=streaming
    )

    ds = ds.map(preprocess_fn, batched=False, remove_columns=['text'])
    
    return ds

# dont have the base model follow the reverse digit instructions
def format_multiplication_example_base_model(example):
    text = example["text"][0] # removing batch dimension

    question = text.split("||")[0].strip()
    num_1 = question.split("*")[0].strip().replace(" ", "")
    num_2 = question.split("*")[1].strip().replace(" ", "")

    # reversing the digits so it's more fair for the base model
    num_1 = int(num_1[::-1])
    num_2 = int(num_2[::-1])

    question = str(num_1) + " * " + str(num_2)

    full_answer = text.split("||")[1].strip()

    return {
        "messages": [
            {
                "role": "system",
                "content": f"Only output one number as your final answer. You MUST not write anything else.",
            },
            {
                "role": "user",
                "content": f"Question: {question}",
            },
            {"role": "assistant", "content": full_answer},
        ]
    }
