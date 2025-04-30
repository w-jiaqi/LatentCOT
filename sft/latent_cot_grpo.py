import sys, os


sys.path.insert(0, os.path.abspath("."))  # hack for imports

from sft.models.latent_tokenizer import LatentTokenizer
from sft.models.latent_cot_model import LatentCOTModel

from data.dataset import get_grpo_dataset
from data.multiplication_dataset import get_4x4_dataset
from data.gsm8k_dataset import get_gsm8k_dataset

from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer
import utils.multiplication_utils as m_utils

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import torch
import argparse
import copy

parser = argparse.ArgumentParser()

parser.add_argument(
    "-m", "--model_pth", type=str, required=True
)

config = parser.parse_args()

wrapper_tokenizer = LatentTokenizer("meta-llama/llama-3.2-1b")
wrapper_model = LatentCOTModel("meta-llama/llama-3.2-1b", tokenizer, freeze_embeddings=True)
wrapper_model.load_state_dict(torch.load(config.model_pth))

model = wrapper_model.model
tokenizer = wrapper_tokenizer._tokenizer
latent_embedding = wrapper_model.latent_embedding
latent_output_embedding = wrapper_model.latent_output_embedding

model = model.to('cuda')
tokenizer = tokenizer.to('cuda')
latent_embedding = latent_embedding.to('cuda')
latent_output_embedding = latent_output_embedding.to('cuda')

original_generate = model.generate

def generate(
    self,
    inputs,
    attention_mask,
    **kwargs
):
    def _expand_token(token_id: int, batch_size: int) -> torch.Tensor: # (batch, 1, dim)
        return self.get_input_embeddings()(
            torch.full((batch_size, 1), token_id, dtype=torch.long, device=self.get_input_embeddings().weight.device)
        )
    
    inputs_embeds = self.get_input_embeddings()(inputs)
    batch_size = inputs.size(0)

    start_latent_col_embed = _expand_token(wrapper_tokenizer.start_latent_id, batch_size)
    end_latent_col_embed = _expand_token(wrapper_tokenizer.end_latent_id, batch_size)
    inputs_embeds = torch.cat((
        inputs_embeds,
        start_latent_col_embed
    ), dim=1)

    attention_mask = torch.cat((
        attention_mask,
        torch.ones((batch_size, 1), device=attention_mask.device)
    ), dim=1)

    kv_cache = None

    for _ in range(8):
        outputs = self(
            inputs_embeds=inputs_embeds if kv_cache is None else inputs_embeds[:, -1:, :],
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=True,
            past_key_values=kv_cache
        )

        kv_cache = outputs.past_key_values

        last_layer = outputs.hidden_states[-1]  # (batch, seq, dim)

        next_embedding = torch.nn.functional.softmax(
            latent_output_embedding(last_layer[:, -1:, :]), dim=-1
        ) @ latent_embedding.weight  # (batch, 1, dim)
        # next_embedding = last_layer[:, -1:, :]

        inputs_embeds = torch.cat((
            inputs_embeds,
            next_embedding
        ), dim=1)
        attention_mask = torch.cat((
            attention_mask,
            torch.ones((batch_size, 1), device=attention_mask.device)
        ), dim=1)

    inputs_embeds = torch.cat((
        inputs_embeds,
        end_latent_col_embed
    ), dim=1)

    attention_mask = torch.cat((
        attention_mask,
        torch.ones((batch_size, 1), device=attention_mask.device)
    ), dim=1)
    
    output = original_generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        **kwargs
    )

    return torch.cat((
        inputs,
        torch.full((batch_size, 1), wrapper_tokenizer.start_latent_id, dtype=torch.long, device=inputs.device),
        torch.full((batch_size, 8), wrapper_tokenizer.latent_id, dtype=torch.long, device=inputs.device),
        torch.full((batch_size, 1), wrapper_tokenizer.end_latent_id, dtype=torch.long, device=inputs.device),
        output
    ), dim=1)

model.generate = generate.__get__(model, type(model))


base_ds = get_4x4_dataset(streaming=False)
dataset = get_grpo_dataset(base_ds)

def reward_ans(prompts, completions, ground_truth, **kwargs):
    ans = [ans.split("<|end-latent|>")[-1] for ans in completions]

    return [m_utils.get_ans_from_response(c) == m_utils.get_ans_from_response(gt) if 1 else -1 for c, gt in zip(ans, ground_truth)]

training_args = GRPOConfig(output_dir="test-grpo", logging_steps=10, beta=0.0)
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_ans,
    args=training_args,
    train_dataset=dataset['train'],
)
trainer.train()