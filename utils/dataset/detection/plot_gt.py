"""
该脚本可以将GroundTruth绘制到原图中，目前GT的类型支持：.txt
"""
import cv2
import os
import sys
sys.path.append(os.getcwd())
from pathlib import Path
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm as TQDM
from prettytable import PrettyTable
from PIL import Image
from typing import Union, Optional, List, Tuple
from utils.general import IMAGE_TYPE
from utils.generator import generator_rgb_colors
from utils.general import xywh2xyxy
from utils.outer import xprint, print_arguments


def get_files(dirpath: str, file_suffixs: Union[List[str], Tuple[str], str]) -> list:
    if not isinstance(file_suffixs, (list, tuple)):
        file_suffixs = [file_suffixs, ]

    dirpath: Path = Path(dirpath)

    if not dirpath.exists():
        print(f"⚠️ The {dirpath.__str__()} is empty!")
        return []
    
    files: list = []
    for suffix in file_suffixs:
        if suffix[0] != '.':
            suffix = '.' + suffix
        files.extend(list(dirpath.rglob(f"*{suffix}")))

    return files


def get_rectangle_coordinates(imgsz: tuple, normal_coordinates) -> list:
    rectangle_coordinates = []
    for xywh in normal_coordinates:
        xyxy = xywh2xyxy(*xywh[1:])
        c = xywh[0]

        # 坐标映射回原图大小
        x1 = round(xyxy[0] * imgsz[0], 4)
        y1 = round(xyxy[1] * imgsz[1], 4)
        x2 = round(xyxy[2] * imgsz[0], 4)
        y2 = round(xyxy[3] * imgsz[1], 4)

        # lt, rb = (x1, y1), (x2, y2)  # left top, right bottom

        rectangle_coordinates.append([c, x1, y1, x2, y2])
    return rectangle_coordinates


def plots(image_filepath: Union[str, Path], points: list, save_dir: Union[str, Path], 
          palette: list, class_names: list=[], thickness=2, save_all: bool=False):
    save_dir: Path = Path(save_dir)

    class_names = list(range(99)) if not class_names else class_names
 
    img = cv2.imread(image_filepath)  # img.shape = (1080, 1920, 3)
    save_filepath = save_dir.joinpath(image_filepath.name)
    
    for c, x1, y1, x2, y2 in points:
        x1, y1 = round(x1), round(y1)
        x2, y2 = round(x2), round(y2)
        cv2.rectangle(
            img=img,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=palette[c],
            thickness=thickness
        )

        cv2.putText(
            img=img, 
            text=f'{c}: {class_names[c]}',
            org=(x1, y1 - 5), 
            fontFace=0, 
            fontScale=0.8, 
            color=palette[c],
            thickness=thickness
        )
    
    cv2.imwrite(save_filepath, img) if points or save_all else ...
        

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--images', type=str, help="The dirpath of images")
    parser.add_argument('--labels', type=str, help="The dirpath of labels, support .txt")
    parser.add_argument('--label_suffix', type=str, default='.txt', help="The suffix of label")
    parser.add_argument('--class_ids', type=int, nargs='+', default=[], help='The specific class will be plot, [] will plot all classes')
    parser.add_argument('--class_names', type=str, nargs='+', default=[], help='The specific class name will be plot, [] will use sequential digits')
    parser.add_argument('--save_dir', type=str, help="The save dirpath of results")
    parser.add_argument('--line_thickness', type=int, default=2, help="The thickness of rectangle")
    parser.add_argument('--save_all', action='store_true', help="The thickness of rectangle")

    args = parser.parse_args()
    return args


def main(args):
    images = get_files(args.images, file_suffixs=IMAGE_TYPE)

    args.save_dir = Path(args.save_dir)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    if args.label_suffix[0] != '.':
        args.label_suffix = '.' + args.label_suffix

    palette = generator_rgb_colors(
            num_color=20,
            return_type='list',
            format_color='rgb',
            rgb2bgr=True
        )
    
    for image_filepath in TQDM(images):
        image_filepath: Path
        label_filepath = Path(args.labels).joinpath(image_filepath.stem + args.label_suffix)

        img = Image.open(image_filepath)
        imgsz = img.size  # W×H
        del img

        with open(label_filepath, 'r') as f:
            objects = f.readlines()
            objects = [c.strip() for c in objects]

        coordinates: list = []
        for obj in objects:
            obj = obj.split(' ')
            obj = list(map(float, obj))  # obj = [2.0, 0.18209617180205415, 0.12488182773109238, 0.09337068160597571, 0.116859243697479]
            obj[0] = int(obj[0])
            obj_class = obj[0]

            if obj_class in args.class_ids:  # 是我们想要的
                coordinates.append(obj)
        
        # 将yolo的xywh变为xyxy且还原为原始尺度
        rectangle_coordinates = get_rectangle_coordinates(imgsz, coordinates)

        # 开始绘制
        plots(image_filepath=image_filepath, points=rectangle_coordinates, 
              save_dir=args.save_dir, palette=palette, class_names=args.class_names, 
              thickness=args.line_thickness, save_all=args.save_all)


if __name__ == "__main__":
    args = get_args()

    print_arguments(argparse=args, wait=True)

    main(args)

    xprint(f"✅ The results save at {args.save_dir}", color='green', bold=True)