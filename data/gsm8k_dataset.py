import sys, os
sys.path.insert(0, os.path.abspath("."))  # hack for imports

from datasets import load_dataset, DatasetDict, Dataset
from typing import Union, Optional
import torch

gsm8k_train_path = "data/gsm8k/train.jsonl"
gsm8k_test_path = "data/gsm8k/test.jsonl"

def get_gsm8k_dataset(streaming: bool) -> Union[DatasetDict, Dataset]:
    def preprocess_fn(example):
        question = example['question']
        full_answer = example['answer']

        reasoning = full_answer.split("\n####")[0].strip()
        answer = full_answer.split("\n####")[1].strip()

        return {
            "question": question,
            "reasoning": reasoning,
            "answer": answer,
        }

    ds = load_dataset(
        "json", 
        data_files={
            "train": gsm8k_train_path,
            "test": gsm8k_test_path,
        },
        streaming=streaming
    )

    ds = ds.map(preprocess_fn, batched=False, remove_columns=['question', 'answer'])
    
    return ds

# def get_gsm8k_dataset(num_train: Optional[int] = None, num_proc: Optional[int] = None) -> Union[DatasetDict, Dataset]:
#     ds_raw = load_dataset("gsm8k", "main")
    
#     if num_train is not None:
#         ds_raw["train"] = ds_raw["train"].select(range(num_train))
    
#     print(f"Subsetting training set to {num_train} examples")
#     def preprocess_fn(example):
#         q = example["question"].strip()
#         a = example["answer"].strip()
        
#         text = q + "||" + a
        
#         question = text.split("||")[0].strip()
#         full_answer = text.split("||")[1].strip()
        
#         if "####" in full_answer:
#             parts = full_answer.split("####", 1)
#             reasoning = parts[0].strip()
#             answer = parts[1].strip()
#         return {
#             "question": question,
#             "reasoning": reasoning,
#             "answer": answer,
#         }
    
#     ds = ds_raw.map(
#         preprocess_fn,
#         batched=False,
#         num_proc=num_proc,
#         remove_columns=ds_raw["train"].column_names,
#     )
    
#     return ds


def format_gsm8k_example_base_model(example):
    """
    Builds a system/user/assistant message format 
    consistent with the new approach in multiplication_dataset.py.
    """
    question = example["question"].strip()
    # If you want chain-of-thought hidden, use only `answer`. 
    # If you want to reveal reasoning, you could combine reasoning + answer.
    final_answer = example["answer"].strip()

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful math problem solver. Output only the final answer."
            },
            {
                "role": "user",
                "content": f"Question: {question}"
            },
            {
                "role": "assistant",
                "content": final_answer
            }
        ]
    }


def get_text_to_latent_dataset(
    tokenizer,
    embedding,
    start_latent_id,
    end_latent_id,
    num_proc=None,
    latent_pool=10,
    num_train=None
):
    ds_raw = load_dataset("gsm8k", "main")

    if num_train is not None:
        ds_raw["train"] = ds_raw["train"].select(range(num_train))

    def build_text(example):
        q = example["question"].strip()
        a = example["answer"].strip()
        return {"text": q + "||" + a}

    ds_with_text = ds_raw.map(
        build_text,
        remove_columns=ds_raw["train"].column_names
    )

    start_latent_embedding = embedding(torch.tensor(start_latent_id))
    end_latent_embedding   = embedding(torch.tensor(end_latent_id))

    def preprocess_fn(examples):
        texts = examples["text"]

        questions    = [t.split("||")[0].strip() for t in texts]
        full_answers = [t.split("||")[1].strip() for t in texts]

        reasonings = []
        for fa in full_answers:
            if "####" in fa:
                reasonings.append(fa.split("####", 1)[0].strip())
            else:
                reasonings.append(fa)

        question_tokens  = tokenizer(
            questions, return_tensors="pt", padding=True, truncation=True, max_length=576
        )
        reasoning_tokens = tokenizer(
            reasonings, return_tensors="pt", padding=True, truncation=True, max_length=576
        )

        question_embeddings  = embedding(question_tokens["input_ids"])      # (B, Q_len, hidden_dim)
        reasoning_embeddings = embedding(reasoning_tokens["input_ids"])     # (B, R_len, hidden_dim)

        batch_size, reasoning_length, latent_dim = reasoning_embeddings.shape

        latent_reasoning_length = (reasoning_length // latent_pool) + 1
        latent_reasoning_embeddings = torch.zeros(
            (batch_size, latent_reasoning_length, latent_dim),
            dtype=reasoning_embeddings.dtype
        )
        for i in range(0, reasoning_length, latent_pool):
            chunk = reasoning_embeddings[:, i : i + latent_pool, :]
            latent_reasoning_embeddings[:, i // latent_pool, :] = chunk.mean(dim=1)

        start_col = start_latent_embedding.expand(batch_size, 1, latent_dim)
        end_col   = end_latent_embedding.expand(batch_size, 1, latent_dim)

        input_embeds = torch.cat(
            (question_embeddings, start_col, latent_reasoning_embeddings, end_col), dim=1
        )

        max_final_len = 576
        final_batch_size, seq_len, embed_dim = input_embeds.shape

        if seq_len < max_final_len:
            pad_amount = max_final_len - seq_len
            pad_tensor = torch.zeros((final_batch_size, pad_amount, embed_dim),
                                     dtype=input_embeds.dtype)
            input_embeds = torch.cat([input_embeds, pad_tensor], dim=1)
        elif seq_len > max_final_len:
            input_embeds = input_embeds[:, :max_final_len, :]

        attention_mask = torch.ones(input_embeds.shape[:-1], dtype=torch.int)

        label_mask = torch.cat(
            (
                torch.zeros(final_batch_size, question_embeddings.shape[1] + 1, dtype=torch.int),
                torch.ones(final_batch_size, latent_reasoning_length + 1, dtype=torch.int)
            ),
            dim=1
        )

        orig_label_mask = label_mask.clone()
        orig_len = orig_label_mask.shape[1]
        if orig_len < max_final_len:
            pad_amount = max_final_len - orig_len
            pad_zeros = torch.zeros((final_batch_size, pad_amount), dtype=orig_label_mask.dtype)
            label_mask = torch.cat([orig_label_mask, pad_zeros], dim=1)
        elif orig_len > max_final_len:
            label_mask = orig_label_mask[:, :max_final_len]
        else:
            label_mask = orig_label_mask

        assert input_embeds.shape[:-1] == label_mask.shape, (
            f"Mismatch in shapes: {input_embeds.shape}, {label_mask.shape}"
        )

        return {
            "input_embeds": input_embeds,
            "attention_mask": attention_mask,
            "label_mask": label_mask
        }

    ds = ds_with_text.map(
        preprocess_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"]
    )

    ds.set_format("pt")
    return ds
