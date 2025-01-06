"""
Author: Le0v1n
Date: 2024/10/09
Description: 根据图片和标签创建空的标签，当前支持标签类型有：YOLO、JSON、XML
Usage: bash data_processing/create_empty_labels.sh
"""
import os
import sys
sys.path.append(os.getcwd())
from pathlib import Path
ROOT = Path.cwd().resolve()
FILE = Path(__file__).resolve()  # 当前脚本的绝对路径
if str(ROOT) not in sys.path:  # 解决VSCode没有ROOT的问题
    sys.path.append(str(ROOT))
ROOT = ROOT.relative_to(Path.cwd())
import threading
from PIL import Image
from typing import Union
from argparse import ArgumentParser, Namespace
from utils.files import get_files
from utils.general import (colorstr, show_args, get_files,  show_args, second_confirm, MyLOGGER, 
                           split_list_equally, exif_size, Recorder, JsonWriter, XMLWriter)
from tqdm import tqdm as TQDM
from utils.items import ImageFormat, LabelTypeMap, LabelFormat


def create_empty_labels(images: list, labels_dirpath: Union[str, Path], 
                        target_dirpath: Union[str, Path], label_type: str) -> None:
    """根据label_type参数从而检查每一张图片是否有标签，如果没有则创建一个空的标签。

    💡 当前支持标签类型有：YOLO、JSON、XML

    Args:
        - images (Path): 存放图片的list
        - labels_dirpath (Path): 标签文件的文件夹路径
        - target_dirpath (Path): 保存结果的文件夹路径。
        - label_type (int): 要生成的标签类型，当前支持：YOLO、JSON、XML
    """
    assert label_type in ('.txt', '.json', '.xml'), f"❌ The label_type only supports ('.txt', '.json', '.xml')!"

    labels_dirpath: Path = Path(labels_dirpath)
    target_dirpath: Path = Path(target_dirpath)

    target_dirpath.mkdir(parents=True, exist_ok=True)

    global pbar
    for image_filepath in images:
        image_filepath = Path(image_filepath)
        
        # 读取图片，并获取图片尺寸和通道数
        im = Image.open(image_filepath)
        img_width, img_height = exif_size(im)
        img_channel = 3 if im.mode == "RGB" else 1 if im.mode == "L" else 4 if im.mode == "RGBA" else "Unknown"
        
        # 确定label位置
        label_filepath = labels_dirpath.joinpath(image_filepath.stem + label_type)
        target_filepath = target_dirpath.joinpath(image_filepath.stem + label_type)
        
        # 判断label是否存在：不存在 -> 负样本
        if label_filepath.exists():  # 如果标签存在 -> 直接跳过
            recorder.positives += 1
        else:  # 标签不存在 -> 创建标签
            LOGGER.silent = True
            LOGGER.info(f"⚠️ [Missing label] The label ({label_filepath.__str__()}) not exist -> {str(image_filepath)}")
            recorder.negatives += 1
            
            # 开始创建负样本
            if label_type == '.txt':  # .txt -> 直接创建空的.txt文件
                if target_filepath.exists():
                    target_filepath.unlink()
                target_filepath.touch(exist_ok=False)
            elif label_type == '.json':  # .json -> 创建空的.json文件且关键k-v要有
                if target_filepath.exists():
                    target_filepath.unlink()
                json_writer = JsonWriter(image_filepath, img_width, img_height)
                json_writer.save(target_filepath)
            elif label_type == '.xml':  # .xml -> 创建空的.xml文件且关键k-v要有
                if target_filepath.exists():
                    target_filepath.unlink()
                xml_writer = XMLWriter(image_filepath, img_width, img_height, img_c=img_channel)
                xml_writer.save(target_filepath)

        pbar.set_description(f"Positives: {recorder.positives} Negatives: {recorder.negatives}")
        pbar.update()


def main(args: Namespace) -> None:
    target_dirpath: Path = Path(args.target_dirpath)

    # ---------- 处理label_type ----------
    assert isinstance(args.label_type, str), f"❌ The parameter of args.label_type must be str instead of {type(args.label_type)}!"
    if args.label_type.lower() in LabelTypeMap:
        args.label_type = LabelTypeMap.get(args.label_type.lower())
    elif args.label_type.lower() in LabelFormat:
        args.label_type = args.label_type.lower()
    else:
        raise KeyError(f"❌ Unkown label type found!")

    # ---------- 获取所有的图片和xml文件 ----------
    images = get_files(Path(args.images_dirpath), file_type=ImageFormat)

    # ---------- 根据线程数，得到每个线程需要处理的图片list ----------
    images_list: list = split_list_equally(images, args.num_threadings)

    # ---------- 开始多线程执行 ----------
    global pbar  # 声明全局变量
    pbar = TQDM(total=len(images), ascii=' >')
    threads = []
    for sub_images in images_list:
        thread = threading.Thread(
            target=create_empty_labels,
            args=(
                sub_images, 
                args.labels_dirpath, 
                args.target_dirpath, 
                args.label_type, 
            )
        )
        threads.append(thread)
        thread.start()

    # 等待所有线程结束
    for thread in threads:
        thread.join()

    # ---------- 检查是否有错误 ----------
    LOGGER.silent = False
    LOGGER.info(show_args(**recorder.get_attributes()))
    errors = recorder.check_attribute(focus=[
        'corrupts',
        'missing_labels',
        'skips',
        'incompeted_points',
        'out_of_boundaries',
        'reversed'
    ])
    LOGGER.error(f"{colorstr('red', 'bold', '❌ Some questions occurred')}: \n{colorstr('red', 'bold', show_args(**errors))}") if errors else ...
    LOGGER.info(f"😄 The results have been saved in {target_dirpath}.")

    
def get_args():
    parser = ArgumentParser()
    parser.add_argument('--images_dirpath', type=str, help='The dirpath of images')
    parser.add_argument('--labels_dirpath', type=str, help='The dirpath of .json files')
    parser.add_argument('--target_dirpath', type=str, help='The dirpath of files (will be saved)')
    parser.add_argument('--label_type', type=str, 
                        choices=['.txt', 'yolo', '.json', 'labelme', '.xml', 'labelImg', 'labelimg'],
                        help='The type of labels will create')
    parser.add_argument("--num_threadings", type=int, default=16, help="使用的线程数，不使用多线程则设置为1")
    
    return parser.parse_args()


if __name__ == "__main__":
    LOGGER = MyLOGGER(FILE, record_level='INFO', silent=False)
    args = get_args()
    LOGGER.info(show_args(args))

    second_confirm(script=FILE, LOGGER=LOGGER)

    # ---------- recoder ----------
    recorder = Recorder()
    recorder.positives = 0
    recorder.negatives = 0
    
    main(args)

    LOGGER.print_logger_filepath()
