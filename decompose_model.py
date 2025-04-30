import sys, os

from sft.models.latent_tokenizer import LatentTokenizer
sys.path.insert(0, os.path.abspath("."))  # hack for imports

from sft.models.latent_cot_model import LatentCOTModel
import utils.utils as utils
import torch
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-m", "--model_pth", type=str, required=True
)
parser.add_argument(
    "-c", "--checkpoints_name", type=str, required=True
)

config = parser.parse_args()

tokenizer = LatentTokenizer("meta-llama/llama-3.2-1b")
model = LatentCOTModel("meta-llama/llama-3.2-1b", tokenizer, freeze_embeddings=True)
model.load_state_dict(torch.load(config.model_pth))

decompose_path = os.path.join(
	"decompose",
	config.checkpoints_name
)

model.model.save_pretrained(os.path.join(decompose_path, "model"))
utils.torch_save(model.latent_embedding, os.path.join(decompose_path, "latent_embedding.pth"))
utils.torch_save(model.latent_output_embedding, os.path.join(decompose_path, "latent_output_embedding.pth"))