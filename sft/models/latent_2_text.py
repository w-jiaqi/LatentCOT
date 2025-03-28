from transformers import AutoModelForCausalLM
from torch import nn
import torch.nn.functional as F

class Latent2Text(nn.Module):
	def __init__(self, model_id, tokenizer):
		super(Latent2Text, self).__init__()

		self.model = AutoModelForCausalLM.from_pretrained(model_id)
		self.tokenizer = tokenizer

		self.model.resize_token_embeddings(len(tokenizer))

	def parameters(self):
		return self.model.parameters()
	
	def embedding(self):
		return self.model.get_input_embeddings()

	def forward(self, input_embeds, attention_mask, labels):
		src_input_embeds = input_embeds[:, :-1, :]
		src_attention_mask = attention_mask[:, :-1]

		outputs = self.model(
			inputs_embeds = src_input_embeds,
			attention_mask = src_attention_mask
		)

		logits = outputs.logits
		pred_logits = logits.view(-1, logits.shape[-1])
		tgt_labels = labels[:, 1:].contiguous().view(-1)

		loss_fn = nn.CrossEntropyLoss()
		loss = loss_fn(pred_logits, tgt_labels)

		return loss
	
	def save_pretrained(self, path):
		self.model.save_pretrained(path)


