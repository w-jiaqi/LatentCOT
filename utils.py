from datetime import datetime
import os


def get_cur_time_string():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def create_dir_from_path(file_path):
    dir_path = os.path.dirname(file_path)

    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
