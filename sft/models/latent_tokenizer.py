from transformers import AutoTokenizer

start_latent_string = "<|start-latent|>"
end_latent_string = "<|end-latent|>"
start_cot_string = "<|start-cot|>"
end_cot_string = "<|end-cot|>"

# wrapper around tokenizer to forward all calls
class LatentTokenizer():
	def __init__(self, tokenizer_path):
		self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

		if not self._tokenizer.convert_tokens_to_ids(start_latent_string):
			self._tokenizer.add_tokens(start_latent_string)

		if not self._tokenizer.convert_tokens_to_ids(end_latent_string):
			self._tokenizer.add_tokens(end_latent_string)

		if not self._tokenizer.convert_tokens_to_ids(start_cot_string):
			self._tokenizer.add_tokens(start_cot_string)

		if not self._tokenizer.convert_tokens_to_ids(end_cot_string):
			self._tokenizer.add_tokens(end_cot_string)

		self.start_latent_id = self._tokenizer.convert_tokens_to_ids(start_latent_string)
		self.end_latent_id = self._tokenizer.convert_tokens_to_ids(end_latent_string)
		self.start_cot_id = self._tokenizer.convert_tokens_to_ids(start_cot_string)
		self.end_cot_id = self._tokenizer.convert_tokens_to_ids(end_cot_string)

	def __getattr__(self, name):
		return getattr(self._tokenizer, name)

 
	# don't really get how these work but i don't think these need to be forwarded

	# def __setattr__(self, name, value):
	# 	if name == '_tokenizer' or hasattr(self, name): # so that __init__ works
	# 		object.__setattr__(self, name, value)
	# 	else:
	# 		setattr(self._tokenizer, name, value)

	# def __dir__(self):
	# 	return list(set(dir(self._tokenizer)) | set(object.__dir__(self)))

	def __len__(self):
		return len(self._tokenizer)

	def __call__(self, *args, **kwds):
		return self._tokenizer(*args, **kwds)
		