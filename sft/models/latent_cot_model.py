import random
import copy
from typing import List, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from sft.models.latent_tokenizer import LatentTokenizer
from enum import Enum, auto
import torch.nn.init as init

class LossType(Enum):
    TOKEN = auto()
    LATENTS = auto()

class LatentCOTModel(nn.Module):
    def __init__(self, model_id: str, tokenizer: LatentTokenizer, freeze_embeddings, tie_weights: bool = False):
        super(LatentCOTModel, self).__init__()

        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.tokenizer = tokenizer

        self.model.resize_token_embeddings(len(tokenizer))
        self.embedding = self.model.get_input_embeddings()
        self.output_embedding = self.model.get_output_embeddings()

        self.latent_embedding = copy.deepcopy(self.embedding)
        self.latent_output_embedding = copy.deepcopy(self.output_embedding)

        if freeze_embeddings:
            self.embedding.requires_grad = False
            self.output_embedding.requires_grad = False

        if tie_weights:
            print("Tying model weights")
            self.model.tie_weights()
    
    def forward(
            self, 
            inputs_embeds: torch.Tensor, 
            attention_mask: torch.Tensor, 
            labels_ids: torch.Tensor, 
            labels_embeds_mask: torch.Tensor,
            output_loss: LossType
    ) -> torch.Tensor:
        if output_loss == LossType.LATENTS:
            return self._calculate_latents_loss(inputs_embeds, attention_mask, labels_embeds_mask)
        elif output_loss == LossType.TOKEN:
            return self._calculate_token_loss(inputs_embeds, attention_mask, labels_ids)

    def _calculate_latents_loss(
            self,
            inputs_embeds: torch.Tensor,
            attention_mask: torch.Tensor,
            labels_embeds_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_latents = inputs_embeds[:, :-1, :].contiguous() # might not be optimal
        tgt_latents = inputs_embeds[:, 1:, :].contiguous()

        src_attention_mask = attention_mask[:, :-1]

        outputs = self.model(
            inputs_embeds=src_latents,
            attention_mask=src_attention_mask,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states[-1]
        
        pred_latents = torch.nn.functional.softmax(self.output_embedding(hidden_states), dim=-1) @ self.embedding.weight

        loss_mask = labels_embeds_mask[:, 1:].contiguous().view(-1)

        latent_dim = pred_latents.shape[-1]

        pred_latents = pred_latents.view(-1, latent_dim)
        pred_latents = pred_latents[loss_mask.bool()]

        tgt_latents = tgt_latents.view(-1, latent_dim)
        tgt_latents = tgt_latents[loss_mask.bool()]

        loss = F.mse_loss(pred_latents, tgt_latents, reduction='none').sum(dim=1).mean(dim=0)

        return loss

    def _calculate_token_loss(
            self,
            inputs_embeds: torch.Tensor,
            attention_mask: torch.Tensor,
            labels_ids: torch.Tensor,
    ) -> torch.Tensor:
        token_loss = self.model(
            inputs_embeds = inputs_embeds,
            attention_mask = attention_mask,
            labels = labels_ids
        ).loss

        return token_loss

    def grpo_forward(
            self,
            question_ids: torch.Tensor,
            question_attention_mask: torch.Tensor,

            reasoning_ids: torch.Tensor,
            reasoning_attention_mask: torch.Tensor,
            reasoning_labels: torch.Tensor,  

            answer_ids: torch.Tensor,
            answer_attention_mask: torch.Tensor,
            answer_labels: torch.Tensor,

            max_new_latents: int,
    ) -> torch.Tensor:
        batch_size = question_ids.size(0)

        base_inputs_embeds, base_attention_mask = self.generate_latents(
            question_ids, question_attention_mask, max_new_latents
        )

        start_cot_col_embed = self._expand_token(self.tokenizer.start_cot_id, batch_size)
        end_cot_col_embed = self._expand_token(self.tokenizer.end_cot_id, batch_size)
        eos_col_embed = self._expand_token(self.tokenizer.eos_token_id, batch_size)

        reasoning_embeds = self.embedding(reasoning_ids)
        answer_embeds = self.embedding(answer_ids)

        ones = torch.ones((batch_size, 1), dtype=base_inputs_embeds.dtype, device=base_attention_mask.device)

        reasoning_inputs_embeds = torch.cat((
            base_inputs_embeds,
            start_cot_col_embed,
            reasoning_embeds,
            end_cot_col_embed,
            eos_col_embed,
        ), dim=1)

        reasoning_attention_mask = torch.cat((
            base_attention_mask,
            ones,  # start_cot token
            reasoning_attention_mask,
            ones,  # end_cot token
            ones   # for eos token
        ), dim=1)

        reasoning_labels = torch.cat((
            torch.full((batch_size, base_inputs_embeds.size(1)), -100, dtype=torch.long, device=reasoning_labels.device),
            torch.full((batch_size, 1), self.tokenizer.start_cot_id, dtype=torch.long, device=reasoning_labels.device),
            reasoning_labels,
            torch.full((batch_size, 1), self.tokenizer.end_cot_id, dtype=torch.long, device=reasoning_labels.device),
            torch.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=torch.long, device=reasoning_labels.device),
        ), dim=1)

        answer_inputs_embeds = torch.cat((
            base_inputs_embeds,
            answer_embeds,
            eos_col_embed,
        ), dim=1)

        answer_attention_mask = torch.cat((
            base_attention_mask,
            answer_attention_mask,
            ones   # eos 
        ), dim=1)

        answer_labels = torch.cat((
            torch.full((batch_size, base_inputs_embeds.size(1)), -100, dtype=torch.long, device=answer_labels.device),
            answer_labels,
            torch.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=torch.long, device=answer_labels.device),
        ), dim=1)

        assert reasoning_inputs_embeds.shape[:-1] == reasoning_attention_mask.shape, (
            f"Mismatch in shapes: {reasoning_inputs_embeds.shape}, {reasoning_attention_mask.shape}"
        )
        assert answer_inputs_embeds.shape[:-1] == answer_attention_mask.shape, (
            f"Mismatch in shapes: {answer_inputs_embeds.shape}, {answer_attention_mask.shape}"
        )
        assert reasoning_inputs_embeds.shape[:-1] == reasoning_labels.shape, (
            f"Mismatch in shapes: {reasoning_inputs_embeds.shape}, {reasoning_labels.shape}"
        )
        assert answer_inputs_embeds.shape[:-1] == answer_labels.shape, (
            f"Mismatch in shapes: {answer_inputs_embeds.shape}, {answer_labels.shape}"
        )

        reasoning_outputs = self.model(
            inputs_embeds=reasoning_inputs_embeds,
            attention_mask=reasoning_attention_mask,
            labels=reasoning_labels
        )

        answer_outputs = self.model(
            inputs_embeds=answer_inputs_embeds,
            attention_mask=answer_attention_mask,
            labels=answer_labels
        )

        loss = reasoning_outputs.loss + 2 * answer_outputs.loss

        return loss

    def generate(
            self, 
            inputs_ids: torch.Tensor,  # (seq_len,) or (batch, seq_len)
            input_attention_mask: torch.Tensor,  # (seq_len,) or (batch, seq_len)
            output_cot: bool,
            max_new_latents: int, 
            max_new_tokens: int, 
            probe_latents: bool = False
    ) -> str:
        if inputs_ids.dim() == 1:
            inputs_ids = inputs_ids.unsqueeze(0)
            input_attention_mask = input_attention_mask.unsqueeze(0)

        batch_dim = inputs_ids.size(0)

        inputs_embeds, attention_mask = self.generate_latents(
            inputs_ids, input_attention_mask, max_new_latents, probe_latents=probe_latents
        )

        if output_cot:
            start_cot_embed = self._expand_token(self.tokenizer.start_cot_id, batch_dim)

            inputs_embeds = torch.cat((inputs_embeds, start_cot_embed), dim=1)

            attention_mask = torch.cat((
                attention_mask,
                torch.ones((batch_dim, 1), dtype=attention_mask.dtype, device=attention_mask.device)
            ), dim=1)

        output = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens
        )

        generated_text = self.tokenizer.decode(output[0])

        return generated_text

    def generate_latents(
            self, 
            inputs_ids: torch.Tensor,  # (batch, seq_len)
            inputs_attention_mask: torch.Tensor,  # (batch, seq_len)
            max_new_latents: int,
            probe_latents: bool = False
    ) -> List[torch.Tensor]:
        batch_size = inputs_ids.size(0)

        inputs_embeds = self.embedding(inputs_ids)   # (batch, seq_len, dim)

        bos_col_embed = self._expand_token(self.tokenizer.bos_token_id, batch_size)
        
        start_latent_col_embed = self._expand_token(self.tokenizer.start_latent_id, batch_size)
        end_latent_col_embed = self._expand_token(self.tokenizer.end_latent_id, batch_size)

        inputs_embeds = torch.cat((
            bos_col_embed,
            inputs_embeds,
            start_latent_col_embed,
        ), dim=1) # (batch, seq, dim)

        attention_mask = torch.cat((
            torch.ones((batch_size, 1), dtype=inputs_attention_mask.dtype, device=inputs_attention_mask.device), # bos
            inputs_attention_mask,
            torch.ones((batch_size, 1), dtype=inputs_attention_mask.dtype, device=inputs_attention_mask.device), # start_latent
        ), dim=1) # (batch, seq)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=inputs_ids.device)
        kv_cache = None

        for _ in range(max_new_latents):
            outputs = self.model(
                inputs_embeds=inputs_embeds if kv_cache is None else inputs_embeds[:, -1:, :],
                attention_mask=attention_mask, # i have absolutely no clue why we can't slice the attention mask here 
                output_hidden_states=True,
                use_cache=True,
                past_key_values=kv_cache
            )

            kv_cache = outputs.past_key_values

			# only will probe the first batch and the top 8 
            if probe_latents:
                logits = outputs.logits  # (batch, seq, vocab)
                topk = torch.topk(torch.softmax(logits[0][-1], dim=0), k=8)
                tokens = [self.tokenizer.decode([token]) for token in topk.indices.tolist()]
                print("       ".join(tokens))

            last_layer = outputs.hidden_states[-1]  # (batch, seq, dim)
            next_prediction = torch.nn.functional.softmax(
                self.latent_output_embedding(last_layer[:, -1, :]), dim=-1
            ) @ self.latent_embedding.weight  # (batch, dim)
            next_prediction = next_prediction.unsqueeze(1)  # (batch, 1, dim)

            mse = torch.nn.functional.mse_loss(
                end_latent_col_embed, next_prediction, reduction="none"
            ).sum(dim=-1).squeeze(-1)  # (batch,)

            finished =  finished | (mse < 1e-3) # we cannot do inplace modifications

            pad_latent = torch.zeros_like(next_prediction)
            pad_attention_mask = torch.zeros_like(attention_mask[:, -1:])

            next_prediction = torch.where(
                finished.view(-1, 1, 1), pad_latent, next_prediction
            )
            next_attention_mask = torch.where(
                finished.view(-1, 1), pad_attention_mask, torch.ones_like(attention_mask[:, -1:])
            )

            inputs_embeds = torch.cat((
                inputs_embeds, 
                next_prediction
            ), dim=1)

            attention_mask = torch.cat((
                attention_mask,
                next_attention_mask
            ), dim=1)

            if finished.all():
                break

        inputs_embeds = torch.cat((
            inputs_embeds,
            end_latent_col_embed,
        ), dim=1)

        attention_mask = torch.cat((
            attention_mask,
            torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device), # end_latent
        ), dim=1)

        assert inputs_embeds.shape[:-1] == attention_mask.shape, (
            f"Mismatch in shapes: {inputs_embeds.shape}, {attention_mask.shape}"
        )

        return (inputs_embeds, attention_mask)
        
    def _expand_token(self, token_id: int, batch_size: int) -> torch.Tensor: # (batch, 1, dim)
        return self.embedding(
            torch.full((batch_size, 1), token_id, dtype=torch.long, device=self.embedding.weight.device)
        )
