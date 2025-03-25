from datasets import load_dataset
import torch

def get_gsm8k_dataset(tokenizer, num_train=None, num_proc=None):
    """
    Loads the official GSM8K dataset from HuggingFace, then pre-processes
    each example has:
        - input_ids
        - attention_mask
        - labels
    """

    ds_raw = load_dataset("gsm8k", "main")  
    if num_train is not None:
        ds_raw["train"] = ds_raw["train"].select(range(num_train))

    # Concatenate "question" and "answer" into a single "text" field for consistency.
    def build_text(example):
        # Some GSM8K "answer" fields include chain-of-thought + final answer. 
        # We'll just store it as-is after "||". 
        # (If the answer has a "####", you can later parse it in a chain-of-thought manner.)
        q = example["question"].strip()
        a = example["answer"].strip()
        return {"text": q + "||" + a}

    ds_with_text = ds_raw.map(
        build_text, 
        remove_columns=ds_raw["train"].column_names
    )

    def preprocess_fn(examples):
        questions = [ex.split("||")[0] + '||' for ex in examples['text']]
        answers = [ex.split("||")[1] for ex in examples['text']]

        questions_tokenized = tokenizer(questions, add_special_tokens=True)
        answers_tokenized = tokenizer(answers, add_special_tokens=False)

        labels = []
        input_ids = []
        attention_mask = []

        for i in range(len(questions)):
            q_ids = questions_tokenized["input_ids"][i]
            a_ids = answers_tokenized["input_ids"][i] + [tokenizer.eos_token_id]

            q_mask = questions_tokenized["attention_mask"][i]
            a_mask = answers_tokenized["attention_mask"][i] + [1]

            lm_labels = ([-100] * len(q_ids)) + a_ids

            labels.append(lm_labels)
            input_ids.append(q_ids + a_ids)
            attention_mask.append(q_mask + a_mask)

        return {
            "labels": labels,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    ds = ds_with_text.map(
        preprocess_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"]
    )

    return ds


def get_text_to_latent_dataset(
    tokenizer,
    embedding,
    start_latent_id,
    end_latent_id,
    num_proc=None,
    latent_pool=10,
    num_train=None
):
    """
    Produces:
        - input_embeds
        - attention_mask
        - label_mask
    for MSE-based 'latent' training.
    """

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
    end_latent_embedding = embedding(torch.tensor(end_latent_id))

    def preprocess_fn(examples):
        texts = examples["text"]
        questions = [t.split("||")[0] + "||" for t in texts]
        full_answers = [t.split("||")[1] for t in texts]

        reasonings = []
        for ans in full_answers:
            parts = ans.split("####")
            if len(parts) > 1:
                # anything before #### is reasoning
                reasonings.append(parts[0].strip())
            else:
                # no ####, so entire answer is reasoning
                reasonings.append(ans.strip())

        question_tokens = tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=576
        )
        reasoning_tokens = tokenizer(
            reasonings,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=576
        )

        question_embeddings = embedding(question_tokens["input_ids"])
        reasoning_embeddings = embedding(reasoning_tokens["input_ids"])

        batch_size, reasoning_length, latent_dim = reasoning_embeddings.shape
        latent_reasoning_length = (reasoning_length // latent_pool) + 1

        latent_reasoning_embeddings = torch.zeros(
            batch_size, latent_reasoning_length, latent_dim
        )

        for i in range(0, reasoning_length, latent_pool):
            chunk = reasoning_embeddings[:, i : i + latent_pool, :]
            latent_reasoning_embeddings[:, i // latent_pool, :] = chunk.mean(dim=1)

        start_latent_col = start_latent_embedding.expand(batch_size, 1, latent_dim)
        end_latent_col = end_latent_embedding.expand(batch_size, 1, latent_dim)

        # Concatenate: question -> <|start-latent|> -> chunked reasoning -> <|end-latent|>
        input_embeds = torch.cat(
            (
                question_embeddings,
                start_latent_col,
                latent_reasoning_embeddings,
                end_latent_col,
            ),
            dim=1,
        )

        attention_mask = torch.ones(input_embeds.shape[:-1])
        label_mask = torch.cat(
            (
                torch.zeros(batch_size, question_embeddings.shape[1] + 1),
                torch.ones(batch_size, latent_reasoning_length + 1),
            ),
            dim=1,
        )

        assert input_embeds.shape[:-1] == label_mask.shape, (
            f"Shapes mismatch: {input_embeds.shape} vs {label_mask.shape}"
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

def format_gsm8k_example_base_model(example):
    question = example["question"].strip()
    answer = example["answer"].strip()

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful math problem solver. Output only the final answer."
            },
            {
                "role": "user",
                "content": f"Question: {question}",
            },
            {
                "role": "assistant",
                "content": answer
            },
        ]
    }
