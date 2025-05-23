import torch
from transformers import AutoModelForCausalLM
from torch import nn
import torch.nn.functional as F
from sft.models.latent_tokenizer import LatentTokenizer

class Text2Latent(nn.Module):
	def __init__(self, model_id: str, tokenizer: LatentTokenizer):
		super(Text2Latent, self).__init__()

		self.model = AutoModelForCausalLM.from_pretrained(model_id)
		self.tokenizer = tokenizer
		# self.model.tie_weights()

		self.model.resize_token_embeddings(len(tokenizer))
		self.embedding = self.model.get_input_embeddings()

	def parameters(self):
		return self.model.parameters()

	def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
		src_embeds = inputs_embeds[:, :-1, :]
		tgt_embeds = inputs_embeds[:, 1:, :] 

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

		loss_func = torch.nn.MSELoss(reduction='none')
		loss = loss_func(masked_preds, masked_targets)
		loss = torch.sum(loss, dim=1)
		loss = torch.mean(loss)

		return loss

	def generate(self, prompt: str, max_new_embeds: int, probe_latents: bool = False) -> torch.Tensor:
		prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
		prompt_embeddings = self.embedding(prompt_ids)

		bos_embedding = self.embedding(torch.tensor(self.tokenizer.bos_token_id))
		bos_embedding_col = bos_embedding.reshape(1, 1, -1) # batch_size * seq_len * latent_dim

		start_latent_embedding = self.embedding(torch.tensor(self.tokenizer.start_latent_id))
		start_latent_embedding_col = start_latent_embedding.reshape(1, 1, -1)

		end_latent_embedding = self.embedding(torch.tensor(self.tokenizer.end_latent_id))
		end_latent_embedding_col = end_latent_embedding.reshape(1, 1, -1)

		inputs_embeds = torch.cat((
			bos_embedding_col,
			prompt_embeddings,
			start_latent_embedding_col,
		), dim=1) # cat along seq dimension

		return_embeds = []

		kv_cache = None 

		for _ in range(max_new_embeds):
			attention_mask = torch.ones(inputs_embeds.shape[:-1])

			outputs = self.model(
				inputs_embeds=inputs_embeds, 
				attention_mask=attention_mask,
				output_hidden_states=True,
				use_cache=True,
				past_key_values=None # TODO: FIX KV CACHE check coconut code
			)

			kv_cache = outputs.past_key_values

			_, greedy_index = torch.max(outputs.logits[0][-1], dim=0) # ignore batch dim, get last prediction

			if probe_latents:
				print(self.tokenizer.decode(greedy_index))

			if greedy_index == self.tokenizer.end_latent_id:
				break

			last_hidden_state = outputs.hidden_states[-1] # hidden states of last layer
			next_prediction = last_hidden_state[0][-1].reshape(1, 1, -1) # ignore batch dim, get hidden state from last token
																	     # then reshape back into batch_dim * seq_len * latent_dim
															
			return_embeds.append(next_prediction)

			inputs_embeds = torch.cat((
				inputs_embeds,
				next_prediction
			), dim=1)

		return torch.cat((
			bos_embedding_col,
			start_latent_embedding_col,
			*return_embeds,
			end_latent_embedding_col
		), dim=1)


	def save_pretrained(self, path: str):
		self.model.save_pretrained(path)
