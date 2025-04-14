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