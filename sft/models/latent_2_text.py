import torch
from torch import nn
import torch.nn.functional as F

class Latent2Text(nn.Module):
	def __init__(self, model, tokenizer):
		super(Latent2Text, self).__init__()

		self.model = model
		self.tokenizer = tokenizer

	def parameters(self):
		return self.model.parameters

	def forward(self, input_embeds, attention_mask, labels):
		# print(input_embeds)
		# print(input_embeds.shape)
		# print(attention_mask)
		# print(attention_mask.shape)
		# print(labels)
		# print(labels.shape)

		src_input_embeds = input_embeds[:, :-1, :]

		outputs = self.model(
			inputs_embeds = src_input_embeds,
			attention_mask = attention_mask
		)

		logits = outputs.logits
		pred_logits = logits.view(-1, logits.shape[-1])
		tgt_labels = labels[:, 1:].contiguous().view(-1)

		# print(pred_logits)
		# print(tgt_labels)

		loss_fn = nn.CrossEntropyLoss()
		loss = loss_fn(pred_logits, tgt_labels)

		return loss


