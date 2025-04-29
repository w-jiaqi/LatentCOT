import sys, os

from sft.models.latent_cot_model import LatentCOTModel
from sft.models.latent_tokenizer import LatentTokenizer

sys.path.insert(0, os.path.abspath("."))  # hack for imports

from data.multiplication_dataset import (
    get_4x4_dataset,
    get_5x5_dataset,
)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

from tqdm.auto import tqdm
import logging

import utils.utils as utils
import utils.multiplication_utils as m_utils

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument(
    "-c", "--config", type=str, required=True
)

config = utils.get_config(parser.parse_args().config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cpu":
    print("WARNING: USING CPU")

# log_file = os.path.join(
#     config.log_dir,
#     f"{utils.string_to_filename(config.base_model)}_{config.dataset}_{utils.get_cur_time_string()}.log",
# )

# utils.create_dir_from_path(log_file)

# logging.basicConfig(filename=log_file, level=logging.INFO)
# logging.getLogger().addHandler(
#     logging.StreamHandler(sys.stdout)
# )  # also print out logs to stdout

tokenizer = LatentTokenizer(config.tokenizer)
model = LatentCOTModel(config.base_model, tokenizer, freeze_embeddings=True).to(device)

if config.model_pth is not None:
    model.load_state_dict(torch.load(config.model_pth))

model.eval()

if config.dataset == "4x4":
    ds = get_4x4_dataset(streaming=False)
elif config.dataset == "5x5":
    ds = get_5x5_dataset(streaming=False)

pb = tqdm(range(len(ds)))

correct = 0

for idx, example in enumerate(ds):
    tokens = tokenizer(example['question'], return_tensors="pt", add_special_tokens=False).to(device)

    ans_ids = model.generate(
        inputs_ids=tokens['input_ids'],
        input_attention_mask=tokens['attention_mask'],
        max_new_latents=config.max_new_latents,
        max_new_tokens=256,
        probe_latents=config.probe_latents,
        output_cot=False,
        unembed_latents=config.unembed_latents,
        dynamically_stop=config.dynamically_stop,
    )

    pred_ans_string = tokenizer.decode(
        ans_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    pred_ans = m_utils.get_ans_from_response(pred_ans_string)
    true_ans = m_utils.get_ans_from_response(example['answer'])

    if pred_ans == true_ans:
        correct += 1

    accuracy = (correct / (idx + 1)) * 100

    logger.info(f"Prompt: {example['question']}\n")
    logger.info(f"Predicted String: {pred_ans_string}\n")
    logger.info(f"True String: {example['answer']}\n")
    logger.info(f"Predicted Answer: {str(pred_ans)}\n")
    logger.info(f"True Answer: {str(true_ans)}\n")
    logger.info(f"Accuracy: {accuracy}%")

    pb.set_description(f"{accuracy}%")
    pb.update(1)
