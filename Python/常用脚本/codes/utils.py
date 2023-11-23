import os
import sys


def create_folder(path, verbose=False):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"文件夹[{path}]不存在，已创建!") if verbose else ...