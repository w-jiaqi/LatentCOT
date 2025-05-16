import sys, os
sys.path.insert(0, os.path.abspath("."))  # hack for imports

from datasets import load_dataset, DatasetDict, Dataset
import torch
from sft.models.latent_tokenizer import LatentTokenizer
from transformers import PreTrainedTokenizerFast
from typing import Union, Optional, Tuple
import math
import torch.nn.functional as F

IGNORE_ID = -100

def compress_embeddings(embeddings: torch.Tensor, latent_pool: int) -> Tuple[int, torch.Tensor]:
    batched = len(embeddings.shape) == 3

    if not batched:
        embeddings = embeddings.unsqueeze(0)
    
    embeddings_reshaped = embeddings.transpose(1, 2)
    
    latent_embeddings = F.avg_pool1d(
        embeddings_reshaped,
        kernel_size=latent_pool,
        stride=latent_pool,
        ceil_mode=True
    )
    
    latent_embeddings = latent_embeddings.transpose(1, 2)
    
    latent_seq_length = latent_embeddings.shape[1]
    
    if not batched:
        latent_embeddings = latent_embeddings.squeeze(0)
    
    return latent_seq_length, latent_embeddings

# labels, input_ids, attention_mask
def get_cot_sft_dataset(dataset: Union[DatasetDict, Dataset], tokenizer: PreTrainedTokenizerFast) -> Union[DatasetDict, Dataset]:
    def preprocess_fn(batch):
        questions = [question + '\n' for question in batch['question']]
        answers = [reasoning + ' #### ' + answer for reasoning, answer in zip(batch['reasoning'], batch['answer'])]

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

    dataset = dataset.map(preprocess_fn, batched=True, remove_columns=dataset['train'].column_names)
    
    return dataset

def get_latent_cot_sft_dataset(
        dataset: Union[DatasetDict, Dataset],
        tokenizer: LatentTokenizer,
        embedding: torch.nn.Module,
        latent_pool: int,
) -> Union[DatasetDict, Dataset]:
    def preprocess_fn(batch):
        torch.set_default_device('cuda')
        start_latent_col = torch.tensor(tokenizer.start_latent_id).unsqueeze(0) # we turn the ids into 1-dimensional tensors so they can be concat'd with other ids
        end_latent_col = torch.tensor(tokenizer.end_latent_id).unsqueeze(0)

        start_cot_col = torch.tensor(tokenizer.start_cot_id).unsqueeze(0)
        end_cot_col = torch.tensor(tokenizer.end_cot_id).unsqueeze(0)

        bos_col = torch.tensor(tokenizer.bos_token_id).unsqueeze(0)
        eos_col = torch.tensor(tokenizer.eos_token_id).unsqueeze(0)

        start_latent_col_embed = embedding(start_latent_col)
        end_latent_col_embed = embedding(end_latent_col)

        start_cot_col_embed = embedding(start_cot_col)
        end_cot_col_embed = embedding(end_cot_col)
        
        bos_col_embed = embedding(bos_col)
        eos_col_embed = embedding(eos_col)
        question = batch['question'][0] # batch size of 1
        reasoning = batch['reasoning'][0]
        answer = batch['answer'][0]

        question_ids = tokenizer.encode(question, return_tensors='pt', add_special_tokens=False)[0] # remove batch dimension
        reasoning_ids = tokenizer.encode(reasoning, return_tensors='pt', add_special_tokens=False)[0]
        answer_ids = tokenizer.encode(answer, return_tensors='pt', add_special_tokens=False)[0]

        question_embeddings = embedding(question_ids)
        reasoning_embeddings = embedding(reasoning_ids)
        answer_embeddings = embedding(answer_ids)

        question_length = question_ids.shape[0]
        reasoning_length = reasoning_ids.shape[0]
        answer_length = answer_ids.shape[0]

        latent_reasoning_length, latent_reasoning_embeddings = compress_embeddings(reasoning_embeddings, latent_pool)

        cot_inputs_embeds = torch.cat((
            bos_col_embed,
            question_embeddings,
            start_latent_col_embed,
            latent_reasoning_embeddings,
            end_latent_col_embed,
            start_cot_col_embed,
            reasoning_embeddings,
            end_cot_col_embed,
            eos_col_embed
        ), dim=0)
        
        ans_inputs_embeds = torch.cat((
            bos_col_embed,
            question_embeddings,
            start_latent_col_embed,
            latent_reasoning_embeddings,
            end_latent_col_embed,
            answer_embeddings,
            eos_col_embed
        ), dim=0)

        cot_attention_mask = torch.ones(cot_inputs_embeds.shape[:-1])
        ans_attention_mask = torch.ones(ans_inputs_embeds.shape[:-1])

        cot_labels = torch.cat((
            torch.full((4 + question_length + latent_reasoning_length, ), IGNORE_ID), # ignore bos, start_latent, end_latent, start_cot
            reasoning_ids,
            end_cot_col,
            eos_col
        ), dim=0)

        ans_labels = torch.cat((
            torch.full((3 + question_length + latent_reasoning_length,), IGNORE_ID), # ignore bos, start_latent, end_latent
            answer_ids,
            eos_col,
        ), dim=0)

        cot_embeds_mask = torch.cat((
            torch.zeros(question_length + 2), # mask loss on bos, start_latent
            torch.ones(latent_reasoning_length + 1), # don't mask loss on end_latent
            torch.zeros(reasoning_length + 3) # mask loss on start_cot, end_cot, eos
        ), dim=0)

        ans_embeds_mask = torch.cat((
            torch.zeros(question_length + 2), # mask loss on bos, start_latent
            torch.ones(latent_reasoning_length + 1), # don't mask loss on end_latent
            torch.zeros(answer_length + 1) # mask loss on eos
        ))

        assert cot_inputs_embeds.shape[0] == cot_labels.shape[0], f"cot_inputs_embeds: {cot_inputs_embeds.shape}, cot_labels: {cot_labels.shape}"
        assert ans_inputs_embeds.shape[0] == ans_labels.shape[0], f"ans_inputs_embeds: {ans_inputs_embeds.shape}, ans_labels: {ans_labels.shape}"

        assert cot_inputs_embeds.shape[0] == cot_embeds_mask.shape[0], f"cot_inputs_embeds: {cot_inputs_embeds.shape}, cot_embeds_mask: {cot_embeds_mask.shape}"
        assert ans_inputs_embeds.shape[0] == ans_embeds_mask.shape[0], f"ans_inputs_embeds: {ans_inputs_embeds.shape}, ans_embeds_mask: {ans_embeds_mask.shape}"

        inputs_embeds = [cot_inputs_embeds, ans_inputs_embeds]
        attention_mask = [cot_attention_mask, ans_attention_mask]
        labels_ids = [cot_labels, ans_labels]
        labels_embeds_mask = [cot_embeds_mask, ans_embeds_mask]

        return {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'labels_ids': labels_ids,
            'labels_embeds_mask': labels_embeds_mask
        }

    dataset = dataset.map(preprocess_fn, batched=True, batch_size=1, with_indices=False, remove_columns=['question', 'reasoning', 'answer'])
    # dataset.set_format('pt')

    return dataset

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

        example['labels_ids'] = torch.cat((
            example['labels_ids'],
            torch.full((max_seq_len - seq_len,), IGNORE_ID) 
        ))

        example['labels_embeds_mask'] = torch.cat((
            example['labels_embeds_mask'],
            torch.zeros((max_seq_len - seq_len))
        ))

    inputs_embeds = torch.stack([example['inputs_embeds'] for example in batch])
    attention_mask = torch.stack([example['attention_mask'] for example in batch])
    labels_ids = torch.stack([example['labels_ids'] for example in batch])
    labels_embeds_mask = torch.stack([example['labels_embeds_mask'] for example in batch])

    return {
        'inputs_embeds': inputs_embeds,
        'attention_mask': attention_mask,
        'labels_ids': labels_ids,
        'labels_embeds_mask': labels_embeds_mask
    }

def get_latent_cot_freeform_dataset(
        dataset: Union[DatasetDict, Dataset],
        tokenizer: LatentTokenizer,
):
    def preprocess_fn(batch):
        question_tokens = tokenizer(batch['question'], add_special_tokens=False)
        reasoning_tokens = tokenizer(batch['reasoning'], add_special_tokens=False)
        answer_tokens = tokenizer(batch['answer'], add_special_tokens=False)

        return {
            'question_ids': question_tokens['input_ids'],
            'question_attention_mask': question_tokens['attention_mask'],

            'reasoning_ids': reasoning_tokens['input_ids'],
            'reasoning_attention_mask': reasoning_tokens['attention_mask'],
            'reasoning_labels': reasoning_tokens['input_ids'],

            'answer_ids': answer_tokens['input_ids'],
            'answer_attention_mask': answer_tokens['attention_mask'],
            'answer_labels': answer_tokens['input_ids'],
        }

    dataset = dataset.map(preprocess_fn, batched=True, remove_columns=['question', 'reasoning', 'answer'])
    dataset.set_format('pt')

    return dataset

def freeform_collate_fn(batch):
    max_question_len = max(example['question_ids'].shape[0] for example in batch)
    max_reasoning_len = max(example['reasoning_ids'].shape[0] for example in batch)
    max_answer_len = max(example['answer_ids'].shape[0] for example in batch)

    for example in batch:
        question_len = example['question_ids'].shape[0]
        reasoning_len = example['reasoning_ids'].shape[0]
        answer_len = example['answer_ids'].shape[0]

		# questions
        example['question_ids'] = torch.cat((
            example['question_ids'], 
            torch.zeros((max_question_len - question_len,), dtype=example['question_ids'].dtype)
        ))
        example['question_attention_mask'] = torch.cat((
            example['question_attention_mask'],
            torch.zeros((max_question_len - question_len,), dtype=example['question_attention_mask'].dtype)
        ))

        # reasoning
        example['reasoning_ids'] = torch.cat((
            example['reasoning_ids'],
            torch.zeros((max_reasoning_len - reasoning_len,), dtype=example['reasoning_ids'].dtype)
        ))
        example['reasoning_attention_mask'] = torch.cat((
            example['reasoning_attention_mask'],
            torch.zeros((max_reasoning_len - reasoning_len,), dtype=example['reasoning_attention_mask'].dtype)
        ))
        example['reasoning_labels'] = torch.cat((
            example['reasoning_labels'],
            torch.full((max_reasoning_len - reasoning_len,), -100, dtype=example['reasoning_labels'].dtype)
        ))

		# answers
        example['answer_ids'] = torch.cat((
            example['answer_ids'],
            torch.zeros((max_answer_len - answer_len,), dtype=example['answer_ids'].dtype)
        ))
        example['answer_attention_mask'] = torch.cat((
            example['answer_attention_mask'],
            torch.zeros((max_answer_len - answer_len,), dtype=example['answer_attention_mask'].dtype)
        ))
        example['answer_labels'] = torch.cat((
            example['answer_labels'],
            torch.full((max_answer_len - answer_len,), -100, dtype=example['answer_labels'].dtype)
        ))

    question_ids = torch.stack([example['question_ids'] for example in batch])
    question_attention_mask = torch.stack([example['question_attention_mask'] for example in batch])

    reasoning_ids = torch.stack([example['reasoning_ids'] for example in batch])
    reasoning_attention_mask = torch.stack([example['reasoning_attention_mask'] for example in batch])
    reasoning_labels = torch.stack([example['reasoning_labels'] for example in batch])

    answer_ids = torch.stack([example['answer_ids'] for example in batch])
    answer_attention_mask = torch.stack([example['answer_attention_mask'] for example in batch])
    answer_labels = torch.stack([example['answer_labels'] for example in batch])

    return {
        'question_ids': question_ids,
        'question_attention_mask': question_attention_mask,

        'reasoning_ids': reasoning_ids,
        'reasoning_attention_mask': reasoning_attention_mask,
        'reasoning_labels': reasoning_labels,

        'answer_ids': answer_ids,
        'answer_attention_mask': answer_attention_mask,
        'answer_labels': answer_labels,
    }


def get_grpo_dataset(
        dataset: Union[DatasetDict, Dataset],
):
    dataset = dataset.rename_column("question", "prompt")
    dataset = dataset.rename_column("answer", "ground_truth")
    dataset = dataset.remove_columns(["reasoning"])

    return dataset

# takes (seq, ) and returns (seq, vocab_size)
def create_soft_labels(token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    labels = torch.zeros((token_ids.shape[0], vocab_size), device=token_ids.device)
    labels.scatter_(1, token_ids.unsqueeze(1), 1.0)

    return labels

def create_latent_embeddings(token_ids: torch.Tensor, latent_pool: int, embedding: torch.nn.Module = None) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len = token_ids.shape[0]
    device = token_ids.device
    pool = latent_pool

    # Pad token_ids so seq_len is divisible by latent_pool, using zeros
    pad_len = (pool - (seq_len % pool)) % pool
    if pad_len > 0:
        padded_token_ids = torch.cat([token_ids, torch.zeros((pad_len,), device=device, dtype=token_ids.dtype)])
    else:
        padded_token_ids = token_ids
    num_latents = padded_token_ids.shape[0] // pool
    tokens_grouped = padded_token_ids.view(num_latents, pool)  # (num_latents, pool)

    # Linearly decaying weights: [pool, ..., 1] / sum
    weights = torch.linspace(pool, 1, steps=pool, device=device)
    weights = weights / weights.sum()  # (pool,)
    weights = weights.unsqueeze(0).expand(num_latents, -1)  # (num_latents, pool)

    # Embedding pooling
    if embedding is not None:
        embeds = embedding(tokens_grouped)  # (num_latents, pool, embed_dim)
        pooled_embeds = (embeds * weights.unsqueeze(-1)).sum(dim=1)  # (num_latents, embed_dim)
    else:
        pooled_embeds = None

    # Soft label pooling
    vocab_size = embedding.weight.shape[0] if embedding is not None else int(token_ids.max().item()) + 1
    soft_labels = torch.zeros(num_latents, vocab_size, device=device)
    soft_labels.scatter_add_(1, tokens_grouped, weights)

    return pooled_embeds, soft_labels

def get_latent_cot_ce_sft_dataset(
        dataset: Union[DatasetDict, Dataset],
        tokenizer: LatentTokenizer,
        embedding: torch.nn.Module,
        latent_pool: int,
) -> Union[DatasetDict, Dataset]:
    device = torch.device('cuda')

    def preprocess_fn(batch):
        start_latent_col = torch.tensor(tokenizer.start_latent_id, device=device).unsqueeze(0) # (1, )
        end_latent_col = torch.tensor(tokenizer.end_latent_id, device=device).unsqueeze(0)

        start_latent_col_embed = embedding(start_latent_col) # (1, embedding_dim)
        end_latent_col_embed = embedding(end_latent_col)

        bos_col = torch.tensor(tokenizer.bos_token_id, device=device).unsqueeze(0)
        eos_col = torch.tensor(tokenizer.eos_token_id, device=device).unsqueeze(0)
        
        bos_col_embed = embedding(bos_col) 
        eos_col_embed = embedding(eos_col)

        question = batch['question']
        reasoning = batch['reasoning']
        answer = batch['answer']

        question_ids = tokenizer.encode(question, return_tensors='pt', add_special_tokens=False)[0] # remove batch dimension
        reasoning_ids = tokenizer.encode(reasoning, return_tensors='pt', add_special_tokens=False)[0]
        answer_ids = tokenizer.encode(answer, return_tensors='pt', add_special_tokens=False)[0]

        question_embeddings = embedding(question_ids) # (seq_len, embedding_dim)
        answer_embeddings = embedding(answer_ids)

        question_length = question_ids.shape[0]

        latent_reasoning_embeddings, latent_reasoning_labels = create_latent_embeddings(reasoning_ids, latent_pool)

        inputs_embeds = torch.cat((
            bos_col_embed,
            question_embeddings,
            start_latent_col_embed,
            latent_reasoning_embeddings,
            end_latent_col_embed,
            answer_embeddings,
            eos_col_embed
        ), dim=0)

        attention_mask = torch.ones(inputs_embeds.shape[:-1], device=device)

        labels = torch.cat((
            torch.zeros((question_length + 2, len(tokenizer)), device=device), # bos, start_latent,
            latent_reasoning_labels,
            create_soft_labels(end_latent_col),
            create_soft_labels(answer_ids),
            create_soft_labels(eos_col)
        ), dim=0)

        return {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'labels': labels,
        }

	# TODO: add batching
    dataset = dataset.map(preprocess_fn, batched=False, with_indices=False, remove_columns=['question', 'reasoning', 'answer'])
    # dataset.set_format('pt')

    return dataset