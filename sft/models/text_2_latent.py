import torch
from torch import nn
import torch.nn.functional as F


class Text2Latent(nn.Module):
	def __init__(self, model, tokenizer):
		super(Text2Latent, self).__init__()

		self.model = model
		self.embedding = model.get_input_embeddings()
		self.tokenizer = tokenizer

	def forward(self, input_embeds, attention_mask, label_mask):
		src_embeds = input_embeds[:, :-1, :]
		tgt_embeds = input_embeds[:, 1:, :] 

		src_attention_mask = attention_mask[:, :-1]

		outputs = self.model(
			inputs_embeds=src_embeds, 
			attention_mask=src_attention_mask, 
			output_hidden_states=True
		)

		pred_embeds = outputs.hidden_states[-1]

		loss_mask = label_mask[:, 1:].contiguous().view(-1)

		_, _, latent_dim = pred_embeds.shape
		pred_embeds = pred_embeds.view(-1, latent_dim)
		tgt_embeds = tgt_embeds.contiguous().view(-1, latent_dim)


		masked_preds = pred_embeds[loss_mask.bool()]
		masked_targets = tgt_embeds[loss_mask.bool()]

		loss = F.mse_loss(masked_preds, masked_targets)

		return loss

