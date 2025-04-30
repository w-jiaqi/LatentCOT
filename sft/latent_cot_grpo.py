import sys, os

sys.path.insert(0, os.path.abspath("."))  # hack for imports

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
    "-m", "--model", type=str, required=True
)
parser.add_argument(
    "-i", "--input_pth", type=str, required=True
)
parser.add_argument(
    "-o", "--output_pth", type=str, required=True
)

config = parser.parse_args()

start_latent_string = "<|start-latent|>"
end_latent_string = "<|end-latent|>"
start_cot_string = "<|start-cot|>"
end_cot_string = "<|end-cot|>"

latent_string = "<|latent-|>"

model_id = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

tokenizer.add_tokens(start_latent_string)
tokenizer.add_tokens(end_latent_string)
tokenizer.add_tokens(start_cot_string)
tokenizer.add_tokens(end_cot_string)
tokenizer.add_tokens(latent_string)

start_latent_id = tokenizer.convert_tokens_to_ids(start_latent_string)
end_latent_id = tokenizer.convert_tokens_to_ids(end_latent_string)
start_cot_id = tokenizer.convert_tokens_to_ids(start_cot_string)
end_cot_id = tokenizer.convert_tokens_to_ids(end_cot_string)
latent_id = tokenizer.convert_tokens_to_ids(latent_string)

model = AutoModelForCausalLM.from_pretrained(config.model)
latent_embedding = copy.deepcopy(model.get_input_embeddings())
latent_output_embedding = copy.deepcopy(model.get_output_embeddings())

model = model.to('cuda')
tokenizer = tokenizer.to('cuda')
latent_embedding = latent_embedding.to('cuda')
latent_output_embedding = latent_output_embedding.to('cuda')

latent_embedding.load_state_dict(torch.load(config.input_pth))
latent_output_embedding.load_state_dict(torch.load(config.output_pth))

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

    start_latent_col_embed = _expand_token(start_latent_id, batch_size)
    end_latent_col_embed = _expand_token(end_latent_id, batch_size)
    eos_token_embed = _expand_token(tokenizer.eos_token_id, batch_size)
    inputs_embeds = torch.cat((
        inputs_embeds,
        start_latent_col_embed
    ), dim=1)

    attention_mask = torch.cat((
        attention_mask,
        torch.ones((batch_size, 1), device=attention_mask.device)
    ), dim=1)

    kv_cache = None

    for _ in range(10):
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
        torch.full((batch_size, 1), start_latent_id, dtype=torch.long, device=inputs.device),
        torch.full((batch_size, 10), latent_id, dtype=torch.long, device=inputs.device),
        torch.full((batch_size, 1), end_latent_id, dtype=torch.long, device=inputs.device),
        output
    ), dim=1)

model.generate = generate.__get__(model, type(model))


base_ds = get_4x4_dataset(streaming=False)
dataset = get_grpo_dataset(base_ds)

# Define the reward function, which rewards completions that are close to 20 characters
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