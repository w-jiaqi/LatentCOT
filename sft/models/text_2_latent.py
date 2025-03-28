from transformers import AutoModelForCausalLM
from torch import nn
import torch.nn.functional as F

class Text2Latent(nn.Module):
	def __init__(self, model_id, tokenizer):
		super(Text2Latent, self).__init__()

		self.model = AutoModelForCausalLM.from_pretrained(model_id)
		self.tokenizer = tokenizer

		self.model.resize_token_embeddings(len(tokenizer))

	def parameters(self):
		return self.model.parameters()

	def embedding(self):
		return self.model.get_input_embeddings()

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

	def save_pretrained(self, path):
		self.model.save_pretrained(path)
