"""

THIS FILE SHOULD BE RAN IN THE PARENT DIRECTORY, NOT INSIDE OF eval/

"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import argparse

from tqdm.auto import tqdm
import logging

logger = logging.getLogger(__name__)

import sys, os

sys.path.insert(0, os.path.abspath("."))  # hack for imports

from data import dataset
import utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct"
)
parser.add_argument("-c", "--finetune_dir", type=str, default=None)
parser.add_argument(
    "-d", "--dataset", choices=["4x4", "5x5", "all"], type=str, required=True
)
parser.add_argument("--device", default="cuda")
parser.add_argument("--dtype", default=torch.bfloat16)
parser.add_argument("--log_dir", default="eval/logs/multiplication")

args = parser.parse_args()

log_file = os.path.join(
    args.log_dir,
    f"{args.base_model_args}-_-{utils.get_cur_time_string()}.log",
)

logging.basicConfig(filename=log_file, level=logging.INFO)
logging.getLogger().addHandler(
    logging.StreamHandler(sys.stdout)
)  # also print out logs to stdout

model = None
tokenizer = None

if args.finetune_dir == None:
    print("USING BASE MODEL")
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

else:
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)

    model = PeftModel.from_pretrained(base_model, args.finetune_dir)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.finetune_dir)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=args.dtype,
    device_map=args.device,
)

ds = None

if args.dataset == "4x4":
    ds = dataset.get_4x4_multiplication_dataset(eval_only=True)


def get_ans_from_response(response):
    answer = (
        response.split("####")[-1].strip().replace(" ", "")[::-1]
    )  # reversing string

    try:
        return int(answer)
    except ValueError:
        return answer


pb = tqdm(range(len(ds)))

correct = 0

for idx, example in enumerate(ds):
    prompt = example["messages"][:-1]

    pred_string = generator(prompt, max_new_tokens=128)[0]["generated_text"][-1][
        "content"
    ]

    true_string = example["messages"][-1]["content"]

    pred_ans = get_ans_from_response(pred_string)
    true_ans = get_ans_from_response(true_string)

    if pred_ans == true_ans:
        correct += 1

    accuracy = (correct / (idx + 1)) * 100

    logger.info(f"Prompt: {prompt}\n")
    logger.info(f"Predicted String: {pred_string}\n")
    logger.info(f"True String: {true_string}\n")
    logger.info(f"Predicted Answer: {str(pred_ans)}\n")
    logger.info(f"True Answer: {str(true_ans)}\n")

    pb.set_description(f"{accuracy}%")
    pb.update(1)

    # if idx % args.logging_steps == 0:
    # 	with open(f"{args.log_dir}/{args.base_model}-{args.finetune_dir}-{args.dataset}.log", "a") as f:
    # 		f.write(f"\nAccuracy: {accuracy}%")
