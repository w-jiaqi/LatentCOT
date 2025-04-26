from datetime import datetime
import os
import re
import torch
import signal
import sys
import yaml
from types import SimpleNamespace

def get_cur_time_string():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# creates the whole directory from a path even if it doesn't exist
def create_dir_from_path(file_path):
    dir_path = os.path.dirname(file_path)

    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


# some strings have bad filenames like meta-llama/Llama-3.2-1B-Instruct, so we just turn this into a safe filename
def string_to_filename(s):
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r'[<>:"|?*]', "_", s)
    s = re.sub(r"[\x00-\x1f\x7f]", "", s)

    return s


def angle_between(v1, v2):
    v1_u = v1 / torch.linalg.norm(v1)
    v2_u = v2 / torch.linalg.norm(v2)

    return torch.rad2deg(torch.arccos(torch.clip(torch.dot(v1_u, v2_u), -1.0, 1.0)))

def torch_save(model, path):
    create_dir_from_path(path)

    torch.save(model.state_dict(), path)

    print(f"Model saved to {path}")

def torch_save_sigint(model, path):
    def handle_sig(sig, frame):
        torch_save(model, path)

        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sig) 

def hf_save_sigint(model, path):
    def handle_sig(sig, frame):
        model.save_pretrained(path)

        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sig) 

def get_config(config_path):
    with open(config_path, "r") as f:
        config = SimpleNamespace(**yaml.safe_load(f))

    return config