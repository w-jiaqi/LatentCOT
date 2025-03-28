from transformers import AutoModelForCausalLM, AutoTokenizer
from data.multiplication_dataset import get_latent_to_text_dataset

model_id = "meta-llama/Llama-3.2-1B"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

embedding = model.get_input_embeddings()

ds = get_latent_to_text_dataset(tokenizer, embedding, 1,2,3,4)