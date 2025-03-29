import sys, os
sys.path.insert(0, os.path.abspath("."))  # hack for imports

from data.dataset import compress_embeddings
from datasets import load_dataset
import torch

TRAIN_4x4_PATH = "data/multiplication/4x4/train.txt"
TEST_4x4_PATH = "data/multiplication/4x4/valid.txt"

IGNORE_ID = -100

# not intended to be used directly
def get_base_dataset(num_train=None, num_proc=None):
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
            "full_answer": full_answer
        }

    ds = load_dataset(
        "text",
        data_files={
            "train": TRAIN_4x4_PATH,
            "test": TEST_4x4_PATH,
        },
    )

    if num_train != None:
        ds["train"] = ds["train"].select(range(num_train))

    ds = ds.map(preprocess_fn, batched=False, num_proc=num_proc, remove_columns=['text'])
    
    return ds

# labels, input_ids, attention_mask
def get_cot_sft_dataset(tokenizer, num_train=None, num_proc=None):
    def preprocess_fn(examples):
        questions = [example.split("||")[0] + '||' for example in examples['text']] # TODO change to \n and then use same base dataset of QRA
        answers = [example.split("||")[1] for example in examples['text']]

        questions_tokenized = tokenizer(questions, add_special_tokens=True) # add begin of seq token
        answers_tokenized = tokenizer(answers, add_special_tokens=False)

        labels = []
        input_ids = []
        attention_mask = []

        for i in range(len(questions)):
            question_input_ids = questions_tokenized['input_ids'][i]
            answer_input_ids = answers_tokenized['input_ids'][i] + [tokenizer.eos_token_id]
            
            question_attention_mask = questions_tokenized['attention_mask'][i]
            answer_attention_mask = answers_tokenized['attention_mask'][i] + [1]
            
            labels.append([IGNORE_ID] * len(question_input_ids) + answer_input_ids) 
            input_ids.append(question_input_ids + answer_input_ids)
            attention_mask.append(question_attention_mask + answer_attention_mask)

        return {'labels': labels, 'input_ids': input_ids, 'attention_mask': attention_mask}

    ds = load_dataset(
        "text",
        data_files={
            "train": TRAIN_4x4_PATH,
            "test": TEST_4x4_PATH,
        },
    )

    if num_train != None:
        ds["train"] = ds["train"].select(range(num_train))

    ds = ds.map(preprocess_fn, batched=True, num_proc=num_proc, remove_columns=['text'])

    return ds

# note: i dont think the way we preprocess here will work for gsm8k because 
# we can't batch process the latents (they will all be different lengths)
def get_text_to_latent_dataset(tokenizer, embedding, start_latent_id, 
                                          end_latent_id, num_proc=None, latent_pool=10, num_train=None):
    start_latent_embedding = embedding(torch.tensor(start_latent_id))
    end_latent_embedding = embedding(torch.tensor(end_latent_id))

    def preprocess_fn(example):
        question = example['question']
        reasoning = example['reasoning']

        question_ids = tokenizer.encode(question, return_tensors="pt", add_special_tokens=True)[0] # remove batch dimension
        reasoning_ids = tokenizer.encode(reasoning, return_tensors="pt", add_special_tokens=False)[0] 

        question_embeddings = embedding(question_ids)
        reasoning_embeddings = embedding(reasoning_ids)

        latent_reasoning_length, latent_reasoning_embeddings = compress_embeddings(reasoning_embeddings, latent_pool)

        start_latent_column = start_latent_embedding.unsqueeze(0)
        end_latent_column = end_latent_embedding.unsqueeze(0)

        inputs_embeds = torch.cat((
            question_embeddings, 
            start_latent_column, 
            latent_reasoning_embeddings, 
            end_latent_column
        ), dim=0)

        attention_mask = torch.ones(inputs_embeds.shape[:-1])
        
        # we mask the loss on the start_latent and don't mask on the end latent
        label_mask = torch.cat((
            torch.zeros(question_embeddings.shape[0] + 1), 
            torch.ones(latent_reasoning_length + 1)
        ), dim=0)

        assert inputs_embeds.shape[:-1] == label_mask.shape, f"inputs_embeds: {inputs_embeds.shape}, label_mask: {label_mask.shape}"

        return {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'label_mask': label_mask
        }

    
    ds = get_base_dataset(num_train, num_proc)

    ds = ds.map(preprocess_fn, batched=False, num_proc=num_proc, remove_columns=ds['train'].column_names)
    ds.set_format("pt")

    return ds

def get_latent_to_text_dataset(tokenizer, embedding, start_latent_id, end_latent_id, 
                               start_cot_id, end_cot_id, num_proc=None, latent_pool = 10, num_train=None):
    start_latent_embedding = embedding(torch.tensor(start_latent_id))
    end_latent_embedding = embedding(torch.tensor(end_latent_id))

    start_cot_embedding = embedding(torch.tensor(start_cot_id))
    end_cot_embedding = embedding(torch.tensor(end_cot_id))

    bos_embedding = embedding(torch.tensor(tokenizer.bos_token_id))
    eos_embedding = embedding(torch.tensor(tokenizer.eos_token_id))

    def preprocess_fn(batch):
        reasoning = batch['reasoning'][0] # batch size of 1
        answer = batch['answer'][0]
        
        reasoning_ids = tokenizer.encode(reasoning, return_tensors="pt", add_special_tokens=False)[0] # remove batch dimension
        answer_ids = tokenizer.encode(answer, return_tensors="pt", add_special_tokens=False)[0]

        reasoning_embeddings = embedding(reasoning_ids) 
        answer_embeddings = embedding(answer_ids)

        latent_reasoning_length, latent_reasoning_embeddings = compress_embeddings(reasoning_embeddings, latent_pool)
        
        cot_inputs_embeds = torch.cat((
            bos_embedding.unsqueeze(0),
            start_latent_embedding.unsqueeze(0), # turning it into 1 x latent_dim
            latent_reasoning_embeddings,
            end_latent_embedding.unsqueeze(0),
            start_cot_embedding.unsqueeze(0),
            reasoning_embeddings,
            end_cot_embedding.unsqueeze(0),
            eos_embedding.unsqueeze(0)
        ), dim=0)

        ans_inputs_embeds = torch.cat((
            bos_embedding.unsqueeze(0),
            start_latent_embedding.unsqueeze(0),
            latent_reasoning_embeddings,
            end_latent_embedding.unsqueeze(0),
            answer_embeddings,
            eos_embedding.unsqueeze(0)
        ), dim=0)

        cot_attention_mask = torch.ones(cot_inputs_embeds.shape[:-1]) # ignore latent_dim
        ans_attention_mask = torch.ones(ans_inputs_embeds.shape[:-1])

        cot_labels = torch.cat((
            torch.full(((4 + latent_reasoning_length,)), IGNORE_ID),  # ignore bos, start_latent, end_latent, and start_cot
            reasoning_ids,
            torch.tensor(end_cot_id).unsqueeze(0),
            torch.tensor(tokenizer.eos_token_id).unsqueeze(0)
        ))

        ans_labels = torch.cat((
            torch.full(((3+latent_reasoning_length,)), IGNORE_ID), # ignore bos, start_latent, end_latent
            answer_ids,
            torch.tensor(tokenizer.eos_token_id).unsqueeze(0)
        ))

        assert cot_inputs_embeds.shape[0] == cot_labels.shape[0]
        assert ans_inputs_embeds.shape[0] == ans_labels.shape[0]

        inputs_embeds = [cot_inputs_embeds, ans_inputs_embeds]
        attention_mask = [cot_attention_mask, ans_attention_mask]
        labels = [cot_labels, ans_labels]

        return {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'labels': labels
        }

    ds = get_base_dataset(num_train, num_proc)

    ds = ds.map(preprocess_fn, batched=True, batch_size=1, num_proc=num_proc, remove_columns=ds['train'].column_names)
    ds.set_format('pt')

    return ds

def collate_fn(batch):
    max_seq_len = max(example['inputs_embeds'].shape[0] for example in batch)
    latent_dim = batch[0]['inputs_embeds'].shape[1]

    for example in batch:
        seq_len = example['inputs_embeds'].shape[0]

        example['inputs_embeds'] = torch.cat((
            example['inputs_embeds'], 
            torch.zeros((max_seq_len - seq_len, latent_dim))
        ))

        example['attention_mask'] = torch.cat((
            example['attention_mask'],
            torch.zeros((max_seq_len - seq_len,))
        ))

        example['labels'] = torch.cat((
            example['labels'],
            torch.full((max_seq_len - seq_len,), IGNORE_ID) 
        ))

    inputs_embeds = torch.stack([example['inputs_embeds'] for example in batch])
    attention_mask = torch.stack([example['attention_mask'] for example in batch])
    labels = torch.stack([example['labels'] for example in batch])

    return {
        'inputs_embeds': inputs_embeds,
        'attention_mask': attention_mask,
        'labels': labels
    }


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
