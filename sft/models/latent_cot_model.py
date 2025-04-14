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
    def __init__(self, model_id: str, tokenizer: LatentTokenizer, tie_weights: bool = False):
        super(LatentCOTModel, self).__init__()

        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.tokenizer = tokenizer

        self.model.resize_token_embeddings(len(tokenizer))
        self.embedding = self.model.get_input_embeddings()
        self.output_embedding = self.model.get_output_embeddings()

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

    def generate(
            self, 
            inputs_ids: torch.Tensor, # (seq_len,)
            output_cot: bool,
            max_new_latents: int, 
            max_new_tokens: int, 
            probe_latents: bool = False
    ) -> str:
        torch.set_default_device('cuda')

        inputs_embeds = self.embedding(inputs_ids) # (seq_len, latent_dim)

        bos_col = torch.tensor(self.tokenizer.bos_token_id).unsqueeze(0)
        bos_col_embed = self.embedding(bos_col)

        start_latent_col = torch.tensor(self.tokenizer.start_latent_id).unsqueeze(0)
        start_latent_col_embed = self.embedding(start_latent_col)

        end_latent_col = torch.tensor(self.tokenizer.end_latent_id).unsqueeze(0)
        end_latent_col_embed = self.embedding(end_latent_col)

        start_cot_col = torch.tensor(self.tokenizer.start_cot_id).unsqueeze(0)
        start_cot_col_embed = self.embedding(start_cot_col)

        inputs_embeds = torch.cat((
            bos_col_embed,
            inputs_embeds,
            start_latent_col_embed,
        ))

        for _ in range(max_new_latents):
            batched_inputs_embeds = inputs_embeds.unsqueeze(0)
            attention_mask = torch.ones(batched_inputs_embeds.shape[:-1])

            outputs = self.model(
                inputs_embeds=batched_inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            if probe_latents:
                sort = torch.sort(torch.softmax(outputs.logits[0][-1], dim=0), descending=True)
                print(sort.values[:8])
                print(repr(self.tokenizer.decode(sort.indices[:8])))
                # print(self.tokenizer.decode(greedy_index))

            hidden_layer = outputs.hidden_states[-1] 
            next_prediction = torch.nn.functional.softmax(self.output_embedding(hidden_layer[0][-1]), dim=0) @ self.embedding.weight
            next_prediction = next_prediction.unsqueeze(0)

            inputs_embeds = torch.cat((
                inputs_embeds,
                next_prediction,
            ), dim=0)


        inputs_embeds = torch.cat((
            inputs_embeds,
            end_latent_col_embed,
        )).unsqueeze(0) # adding batch dim

        if output_cot:
            inputs_embeds = torch.cat((
                inputs_embeds,
                start_cot_col_embed.unsqueeze(0),
            ), dim=1) # along seq dim

        attention_mask = torch.ones(inputs_embeds.shape[:-1])

        output = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens
        )
        
        generated_text = self.tokenizer.decode(output[0]) # removing batch dim

        return generated_text

    def save_pretrained(self, path: str):
        self.model.save_pretrained(path)

