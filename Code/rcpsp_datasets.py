from __future__ import annotations

import os

path_to_data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data/rcpsp/")
files_available = [os.path.join(path_to_data, f) for f in os.listdir(path_to_data)]


def get_data_available():
    files = [f for f in os.listdir(path_to_data) if "pk" not in f and "json" not in f]
    return [os.path.join(path_to_data, f) for f in files]


def get_complete_path(root_path: str) -> str:  # example root_path="j101.sm"
    path_list = [f for f in get_data_available() if root_path in f]
    if len(path_list) > 0:
        return path_list[0]
    return None
