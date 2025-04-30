import sys, os
import copy
import re  # added for regex matching

sys.path.insert(0, os.path.abspath("."))  # hack for imports

from sft.models.latent_tokenizer import LatentTokenizer
from sft.models.latent_cot_model import LatentCOTModel

from data.dataset import get_grpo_dataset
from data.multiplication_dataset import get_4x4_dataset
from data.gsm8k_dataset import get_gsm8k_dataset

from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer
import utils.multiplication_utils as m_utils

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, PPOConfig, PPOTrainer
import torch
import argparse

print("DONT FORGET TO ADD A CONFIG TO SET THE LATENT COUNT")
print("also figure out how to update input/output latent embeds")
LATENT_COUNT = 8

parser = argparse.ArgumentParser()

parser.add_argument(
    "-m", "--model_pth", type=str, required=True
)

config = parser.parse_args()

wrapper_tokenizer = LatentTokenizer("meta-llama/llama-3.2-1b")
wrapper_model = LatentCOTModel("meta-llama/llama-3.2-1b", wrapper_tokenizer, freeze_embeddings=True)
wrapper_model.load_state_dict(torch.load(config.model_pth))

model = wrapper_model.model
tokenizer = wrapper_tokenizer._tokenizer
latent_embedding = wrapper_model.latent_embedding
latent_output_embedding = wrapper_model.latent_output_embedding

# model = model.to('cuda')
# tokenizer = tokenizer.to('cuda')
latent_embedding = latent_embedding.to('cuda')
latent_output_embedding = latent_output_embedding.to('cuda')

tokenizer.add_tokens("<|latent|>")
latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")

model.resize_token_embeddings(len(tokenizer))

original_generate = model.generate

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print(model.get_input_embeddings().requires_grad)
print(model.get_output_embeddings().requires_grad)

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

    bos_col_embed = _expand_token(tokenizer.bos_token_id, batch_size)
    start_latent_col_embed = _expand_token(wrapper_tokenizer.start_latent_id, batch_size)
    end_latent_col_embed = _expand_token(wrapper_tokenizer.end_latent_id, batch_size)

    inputs_embeds = torch.cat((
        bos_col_embed,
        inputs_embeds,
        start_latent_col_embed
    ), dim=1)

    attention_mask = torch.cat((
        torch.ones((batch_size, 1), device=attention_mask.device),
        attention_mask,
        torch.ones((batch_size, 1), device=attention_mask.device)
    ), dim=1)

    kv_cache = None

    for _ in range(LATENT_COUNT):
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
            torch.ones((batch_size, 1), dtype=torch.float32, device=attention_mask.device)
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
        max_new_tokens=30,
        **kwargs
    )

    return torch.cat((
        inputs,
        torch.full((batch_size, 1), wrapper_tokenizer.start_latent_id, dtype=torch.long, device=inputs.device),
        torch.full((batch_size, LATENT_COUNT), latent_id, dtype=torch.long, device=inputs.device),
        torch.full((batch_size, 1), wrapper_tokenizer.end_latent_id, dtype=torch.long, device=inputs.device),
        output
    ), dim=1)

model.generate = generate.__get__(model, type(model))


base_ds = get_4x4_dataset(streaming=False)
dataset = get_grpo_dataset(base_ds)

def reward_ans(prompts, completions, ground_truth, **kwargs):
    print(prompts)
    print(completions)
    ans = [ans.split("<|end-latent|>")[-1] for ans in completions]
    print(ans)

    rewards = []

    for c, gt in zip(ans, ground_truth):
        pred_val = m_utils.get_ans_from_response(c)
        true_val = m_utils.get_ans_from_response(gt)

        if true_val is None or pred_val is None:
            reward = -2
        else:
            pred_str = str(pred_val)
            true_str = str(true_val)
            # Compare digit by digit
            correct = sum(1 for pd, td in zip(pred_str, true_str) if pd == td)
            reward = correct / len(true_str)
            
            # Bonus reward if c is formatted exactly as "digit digit ... digit" with exactly 8 digits
            # E.g., "4 2 8 1 6 2 1 4" (digits can be different)
            if re.fullmatch(r'[0-9]( [0-9]){7}', c.strip()):
                reward += 0.5

        rewards.append(reward)

    return rewards

class RewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, prompts, completions, ground_truth, **kwargs):
        # Extract answers after the latent token
        ans = [ans.split("<|end-latent|>")[-1] for ans in completions]
        rewards = []
        for c, gt in zip(ans, ground_truth):
            pred_val = m_utils.get_ans_from_response(c)
            true_val = m_utils.get_ans_from_response(gt)
            if true_val is None or pred_val is None:
                reward = -2.0
            else:
                pred_str = str(pred_val)
                true_str = str(true_val)
                # Compare digit by digit
                correct = sum(1 for pd, td in zip(pred_str, true_str) if pd == td)
                reward = correct / len(true_str)
                # Bonus reward if c is formatted as "digit digit ... digit" with exactly 8 digits
                if re.fullmatch(r'[0-9]( [0-9]){7}', c.strip()):
                    reward += 0.5
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)

reward_model = RewardModel()

# training_args = PPOConfig(output_dir="test-grpo", logging_steps=10, beta=0.0)
training_args = PPOConfig(output_dir="test-grpo", logging_steps=10)
trainer = PPOTrainer(
    model=model,
    ref_model=copy.deepcopy(model),
    processing_class=tokenizer,
    # reward_funcs=reward_ans,
    reward_model=reward_model,
    args=training_args,
    train_dataset=dataset['train'],
)
trainer.train()