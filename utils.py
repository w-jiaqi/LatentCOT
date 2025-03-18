from datetime import datetime
import os
import re


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
