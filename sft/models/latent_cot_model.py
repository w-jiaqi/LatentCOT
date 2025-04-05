from collections import namedtuple
from typing import List, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from sft.models.latent_tokenizer import LatentTokenizer

Output = namedtuple('Output', ['token_loss', 'latents_loss'])

class LatentCOTModel(nn.Module):
	def __init__(self, model_id: str, tokenizer: LatentTokenizer):
		super(LatentCOTModel, self).__init__()

		self.model = AutoModelForCausalLM.from_pretrained(model_id)
		self.tokenizer = tokenizer

		self.model.resize_token_embeddings(len(tokenizer))
		self.embedding = self.model.get_input_embeddings()

	def parameters(self):
		return self.model.parameters()
	
	def forward(
			self, 
			inputs_embeds: torch.Tensor, 
			attention_mask: torch.Tensor, 
			labels_ids: torch.Tensor, 
			labels_embeds_mask: torch.Tensor
	) -> Output:
		token_loss = self._calculate_token_loss(inputs_embeds, attention_mask, labels_ids)
		latents_loss = self._calculate_latents_loss(inputs_embeds, attention_mask, labels_embeds_mask)

		return Output(token_loss=token_loss, latents_loss=latents_loss)

	def _calculate_latents_loss(
			self,
			inputs_embeds: torch.Tensor,
			attention_mask: torch.Tensor,
			labels_embeds_mask: torch.Tensor,
	) -> int:
		src_latents = inputs_embeds[:, :-1, :].contiguous() # might not be optimal
		tgt_latents = inputs_embeds[:, 1:, :].contiguous()

		src_attention_mask = attention_mask[:, :-1]

		outputs = self.model(
			inputs_embeds=src_latents,
			attention_mask=src_attention_mask,
			output_hidden_states=True
		)

		pred_latents = outputs.hidden_states[-1]

		loss_mask = labels_embeds_mask[:, 1:].contiguous().view(-1)

		latent_dim = pred_latents.shape[-1]

		pred_latents = pred_latents.view(-1, latent_dim)
		pred_latents = pred_latents[loss_mask.bool()]

		tgt_latents = tgt_latents.view(-1, latent_dim)
		tgt_latents = tgt_latents[loss_mask.bool()]

		loss = F.mse_loss(pred_latents, tgt_latents, reduction='mean')

		return loss

	def _calculate_token_loss(
			self,
			inputs_embeds: torch.Tensor,
			attention_mask: torch.Tensor,
			labels_ids: torch.Tensor,
	) -> int:
		token_loss = self.model(
			inputs_embeds = inputs_embeds,
			attention_mask = attention_mask,
			labels = labels_ids
		).loss

		return token_loss

	def generate(
			self, 
			inputs_ids: List[int], 
			max_new_latents: int, 
			max_new_tokens: int, 
			probe_model: bool = False
	) -> str:
		pass
		

	def save_pretrained(self, path: str):
		self.model.save_pretrained(path)

