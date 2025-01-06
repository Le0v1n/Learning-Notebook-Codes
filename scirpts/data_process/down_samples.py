"""
Author: Le0v1n
Date: 2024/12/30
Description: 对某个文件夹中的文件进行下采样
Usage: bash scirpts/data_process/down_samples.sh
"""
import os
import sys
sys.path.append(os.getcwd())
import random
from pathlib import Path
ROOT = Path.cwd().resolve()
FILE = Path(__file__).resolve()
sys.path.append(str(ROOT)) if str(ROOT) not in sys.path else ...
ROOT = ROOT.relative_to(Path.cwd())
from argparse import ArgumentParser, Namespace
from utils.files import get_files
from utils.general import show_args
from tqdm import tqdm as TQDM


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--src_dirpath', type=str, help='The dirpath of source')
    parser.add_argument('--ratio', type=float, default=1.0, help='Down-sample ratio, it must be in [0.0, 1.0]')
    parser.add_argument('--file_types', type=str, nargs='+', help='The file types will be down_samples')
    
    return parser.parse_args()


def main(args: Namespace):
    assert 0 <= args.ratio <= 1, f"ERROR ❌ The ratio of down-sample must be in [0.0, 1.0] not {args.ratio}"

    # get all files
    files: list = get_files(args.src_dirpath, file_type=args.file_types)

    # shuffle
    random.shuffle(files)

    # truncation
    n_origin: int = len(files)
    n_retained: int = int(n_origin * args.ratio)
    files: list = files[n_retained + 1: ]

    # delete redundant files
    for filepath in TQDM(files, ascii='> '):
        filepath: Path
        filepath.unlink()

    print(f"{n_origin - n_retained} files have been deleted, results saved to {args.src_dirpath}")
    

if __name__ == '__main__':
    args = parse_args()

    print(show_args(args))

    main(args)