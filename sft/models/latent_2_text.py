import torch
from transformers import AutoModelForCausalLM
from torch import nn
from sft.models.latent_tokenizer import LatentTokenizer

class Latent2Text(nn.Module):
	def __init__(self, model_id: str, tokenizer: LatentTokenizer):
		super(Latent2Text, self).__init__()

		self.model = AutoModelForCausalLM.from_pretrained(model_id)
		self.tokenizer = tokenizer

		self.model.resize_token_embeddings(len(tokenizer))
		self.embedding = self.model.get_input_embeddings()

	def parameters(self):
		return self.model.parameters()
	
	def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		src_inputs_embeds = inputs_embeds[:, :-1, :]
		src_attention_mask = attention_mask[:, :-1]

		outputs = self.model(
			inputs_embeds = src_inputs_embeds,
			attention_mask = src_attention_mask
		) # TODO: try just setting labels = labels and returning outputs.loss

		logits = outputs.logits
		pred_logits = logits.view(-1, logits.shape[-1])
		tgt_labels = labels[:, 1:].contiguous().view(-1)

		loss_fn = nn.CrossEntropyLoss()
		loss = loss_fn(pred_logits, tgt_labels)

		return loss

	# inputs_embeds should be batch_dim * seq_len * latent_dim
	def generate(self, inputs_embeds: torch.Tensor, output_cot=False, max_new_tokens: int=128) -> str:
		if output_cot:
			inputs_embeds = torch.cat((
				inputs_embeds,
				self.embedding(torch.tensor(self.tokenizer.start_cot_id)).reshape(1,1,-1)
			), dim=1)

		attention_mask = torch.ones(inputs_embeds.shape[:-1])

		output = self.model.generate(
			inputs_embeds=inputs_embeds, 
			attention_mask=attention_mask, 
			max_new_tokens=max_new_tokens
		)

		generated_text = self.tokenizer.decode(output[0]) # removing batch_dim

		return generated_text
	
	def save_pretrained(self, path: str):
		self.model.save_pretrained(path)