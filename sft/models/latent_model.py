from torch import nn

class LatentModel():
	def __init__(self, text_to_latent, latent_to_text):
		self.text_to_latent = text_to_latent
		self.latent_to_text = latent_to_text

	def generate(self, prompt):
		pass
