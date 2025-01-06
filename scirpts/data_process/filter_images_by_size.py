"""
Author: Le0v1n
Date: 2024/12/30
Description: 通过图片尺寸大小对图片进行过滤
Usage: bash scirpts/data_process/down_samples.sh
"""
import os
import sys
sys.path.append(os.getcwd())
from pathlib import Path
ROOT = Path.cwd().resolve()
FILE = Path(__file__).resolve()
sys.path.append(str(ROOT)) if str(ROOT) not in sys.path else ...
ROOT = ROOT.relative_to(Path.cwd())
from argparse import ArgumentParser, Namespace
from utils.files import get_files
from utils.general import show_args, verify_image
from utils.items import ImageFormat
from tqdm import tqdm as TQDM
from PIL import Image



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--src_dirpath', type=str, help='The dirpath of source')
    parser.add_argument('--imgsz', type=int, nargs='+', default=640, help='The image size for filter (H, W)')
    parser.add_argument("--num_threadings", type=int, default=16)
    
    return parser.parse_args()


def main(args: Namespace):
    # check args.imgsz
    if isinstance(args.imgsz, int):
        imgsz = [args.imgsz] * 2
    elif isinstance(args.imgsz, (list, tuple)) and 1 <= len(args.imgsz) <= 2:
        imgsz = [int(e) for e in args.imgsz] if len(args.imgsz) == 2 else [int(args.imgsz[0]), int(args.imgsz[0])]
    else:
        raise ValueError(f"ERROR ❌ Unsupport type found -> {type(imgsz).__name__}")
    
    # get all files
    images: list = get_files(args.src_dirpath, file_type=ImageFormat)

    # interate image and filter
    recorder: dict = {
        'retained': 0,
        'deleted': 0,
    }
    pbar = TQDM(total=len(images), ascii='> ')
    for image_filepath in images:
        try:
            # Open the image and get width & height
            img = Image.open(image_filepath)
            w, h = img.size

            # filter
            if h < imgsz[0] or w < imgsz[1]:
                image_filepath.unlink()  # delete image file
                recorder['deleted'] += 1
            else:
                recorder['retained'] += 1
        except Exception as e:
            print(f"ERROR ❌ Exception found: {e}")
            image_filepath.unlink()
        finally:
            pbar.set_description(f"Retained: {recorder['retained']}  Deleted: {recorder['deleted']}")
            pbar.update()
    pbar.close()

if __name__ == '__main__':
    args = parse_args()

    print(show_args(args))

    main(args)