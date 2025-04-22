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
    def __init__(self, model_id: str, tokenizer: LatentTokenizer, tie_weights: bool = False):
        super(LatentCOTModel, self).__init__()

        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.tokenizer = tokenizer

        self.model.resize_token_embeddings(len(tokenizer))
        self.embedding = self.model.get_input_embeddings()
        self.output_embedding = self.model.get_output_embeddings()

        self.latent_embedding = nn.Embedding(
            len(tokenizer),
            self.embedding.embedding_dim,
        )

        self.latent_embedding = copy.deepcopy(self.embedding)
        self.latent_output_embedding = copy.deepcopy(self.output_embedding)

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
            question_ids: torch.Tensor,             # (batch, q_seq)
            question_attention_mask: torch.Tensor,  # (batch, q_seq)
            reasoning_ids: torch.Tensor,            # (batch, r_seq)
            reasoning_attention_mask: torch.Tensor, # (batch, r_seq)
            answer_ids: torch.Tensor,               # (batch, a_seq)
            answer_attention_mask: torch.Tensor,    # (batch, a_seq)
            max_new_latents: int,
    ) -> torch.Tensor:
        batch_size = question_ids.size(0)

        question_embeds = self.embedding(question_ids)   # (batch, q_seq, dim)
        reasoning_embeds = self.embedding(reasoning_ids) # (batch, r_seq, dim)
        answer_embeds = self.embedding(answer_ids)       # (batch, a_seq, dim)

        # Prepare special token ids and embeddings, expand for batch
        def expand_token(token_id):
            return self.embedding(
                torch.full((batch_size, 1), token_id, dtype=question_ids.dtype, device=question_ids.device)
            ) # (batch, 1, dim)

        bos_col_embed = expand_token(self.tokenizer.bos_token_id)
        eos_col_embed = expand_token(self.tokenizer.eos_token_id)
        start_latent_col_embed = expand_token(self.tokenizer.start_latent_id)
        end_latent_col_embed = expand_token(self.tokenizer.end_latent_id)
        start_cot_col_embed = expand_token(self.tokenizer.start_cot_id)
        end_cot_col_embed = expand_token(self.tokenizer.end_cot_id)

        # Build initial input embeds: [BOS] + question + [START_LATENT]
        inputs_embeds = torch.cat((
            bos_col_embed,
            question_embeds,
            start_latent_col_embed,
        ), dim=1) # (batch, seq, dim)

        # Build initial attention mask using question_attention_mask
        # [BOS] (1), question_attention_mask, [START_LATENT] (1)
        attention_mask = torch.cat((
            torch.ones((batch_size, 1), dtype=question_attention_mask.dtype, device=question_attention_mask.device), # bos
            question_attention_mask,
            torch.ones((batch_size, 1), dtype=question_attention_mask.dtype, device=question_attention_mask.device), # start_latent
        ), dim=1) # (batch, seq)

        # finished = torch.zeros(batch_size, dtype=torch.bool, device=question_ids.device)
        # kv_cache = None

        for _ in range(max_new_latents):
            # if kv_cache is None:
            #     model_inputs = inputs_embeds
            # else:
            #     model_inputs = inputs_embeds[:, -1:, :]
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=True,
                # past_key_values=kv_cache
            )

            # kv_cache = outputs.past_key_values

            hidden_layer = outputs.hidden_states[-1]  # (batch, seq, dim)
            next_prediction = torch.nn.functional.softmax(
                self.latent_output_embedding(hidden_layer[:, -1]), dim=-1
            ) @ self.latent_embedding.weight  # (batch, dim)
            next_prediction = next_prediction.unsqueeze(1)  # (batch, 1, dim)

            # # Check for end_latent token for each batch element
            # mse = torch.nn.functional.mse_loss(
            #     end_latent_col_embed, next_prediction, reduction="none"
            # ).sum(dim=-1).squeeze(-1)  # (batch, 1) -> (batch,)
            # newly_finished = (mse < 1e-3) & (~finished)
            # finished = finished | newly_finished

            # # For finished elements, pad with zeros; for unfinished, use next_prediction
            # pad_latent = torch.zeros_like(next_prediction)
            # next_prediction = torch.where(
            #     finished.view(-1, 1, 1), pad_latent, next_prediction
            # )

            inputs_embeds = torch.cat((inputs_embeds, next_prediction), dim=1)
            attention_mask = torch.cat((
                attention_mask,
                torch.zeros((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
            ), dim=1)

            # # If all finished, break
            # if finished.all():
            #     break

        # Build reasoning and answer input embeds
        reasoning_inputs_embeds = torch.cat((
            inputs_embeds,
            end_latent_col_embed,
            start_cot_col_embed,
            reasoning_embeds,
            end_cot_col_embed,
            eos_col_embed,
        ), dim=1) # (batch, seq, dim)

        reasoning_attention_mask = torch.cat((
            attention_mask,
            torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device), # end_latent
            torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device), # start_cot
            reasoning_attention_mask,
            torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device), # end_cot
            torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device), # eos
        ), dim=1)

        answer_inputs_embeds = torch.cat((
            inputs_embeds,
            end_latent_col_embed,
            answer_embeds,
            eos_col_embed,
        ), dim=1) # (batch, seq, dim)

        answer_attention_mask = torch.cat((
            attention_mask,
            torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device), # end_latent
            answer_attention_mask,
            torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device), # eos
        ), dim=1)

        # Build labels for reasoning and answer
        # -100 for non-label positions, then reasoning_ids, end_latent, start_cot
        reasoning_label_prefix_len = reasoning_inputs_embeds.shape[1] - (reasoning_embeds.shape[1] + 2)
        answer_label_prefix_len = answer_inputs_embeds.shape[1] - (answer_embeds.shape[1] + 1)

        reasoning_labels = torch.cat((
            torch.full((batch_size, reasoning_label_prefix_len), -100, dtype=reasoning_ids.dtype, device=reasoning_ids.device),
            reasoning_ids,
            torch.full((batch_size, 1), self.tokenizer.end_cot_id, dtype=reasoning_ids.dtype, device=reasoning_ids.device),
            torch.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=reasoning_ids.dtype, device=reasoning_ids.device),
        ), dim=1)

        answer_labels = torch.cat((
            torch.full((batch_size, answer_label_prefix_len), -100, dtype=answer_ids.dtype, device=answer_ids.device),
            answer_ids,
            torch.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=answer_ids.dtype, device=answer_ids.device),
        ), dim=1)

        assert reasoning_labels.shape == reasoning_inputs_embeds.shape[:-1], (
            f"Mismatch in shapes: {reasoning_labels.shape}, {reasoning_inputs_embeds.shape[:2]}"
        )
        assert answer_labels.shape == answer_inputs_embeds.shape[:-1], (
            f"Mismatch in shapes: {answer_labels.shape}, {answer_inputs_embeds.shape[:2]}"
        )

        assert reasoning_attention_mask.shape == reasoning_inputs_embeds.shape[:-1], (
            f"Mismatch in shapes: {reasoning_attention_mask.shape}, {reasoning_inputs_embeds.shape[:2]}"
        )

        assert answer_attention_mask.shape == answer_inputs_embeds.shape[:-1], (
            f"Mismatch in shapes: {answer_attention_mask.shape}, {answer_inputs_embeds.shape[:2]}"
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

        reasoning_loss = reasoning_outputs.loss
        answer_loss = answer_outputs.loss

        loss = reasoning_loss + answer_loss

        return loss

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

            if (torch.nn.functional.mse_loss(end_latent_col_embed, next_prediction, reduction="sum") < 1e-3).all():
                break

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

