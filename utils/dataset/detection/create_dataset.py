__doc__ = """
- 功能：生成数据集（支持VOC2007、VOC2012、COCO）
- 数据集格式：
    - Datasets/VOC2012
        VOCtrainval_11-May-2012/VOCdevkit/VOC2012
        ├── Annotations  # 存放所有的标签
        │   ├── 2010_003107.json
        │   ├── 2010_003108.json
        │   ├── ...
        ├── JPEGImages  # 存放所有的图片
        │   ├── 2010_003107.jpg
        │   ├── 2010_003108.jpg
        │   ├── ...
        └─ ImageSets
            └─ Main  # 定义哪些脚本用来训练、验证、测试
                ├── train.txt
                ├── val.txt
                └── test.txt
    
    - Datasets/COCO
        ├── train
        │   ├── images
        │   │   ├── 000000000061.jpg
        │   │   ├── 000000000071.jpg
        │   │   ├── ...
        │   └── labels
        │       ├── 000000000061.txt
        │       ├── 000000000071.txt
        │       ├── ...
        └── val
        │   ├── images
        │   │   ├── 000000000009.jpg
        │   │   ├── 000000000025.jpg
        │   │   ├── ...
        │   └── labels
        │       ├── 000000000009.txt
        │       ├── 000000000025.txt
        │       ├── ...
        └── test
            ├── images
            │   ├── 000000000009.jpg
            │   ├── 000000000025.jpg
            │   ├── ...
            └── labels
                ├── 000000000009.txt
                ├── 000000000025.txt
                ├── ...
"""


import sys
import argparse
import threading
from pathlib import Path
import random
import math
from PIL import Image
import time
try:
    from tqdm.rich import tqdm
except:
    from tqdm import tqdm


ROOT = Path.cwd().resolve()
FILE = Path(__file__).resolve()  # 当前脚本的绝对路径
if str(ROOT) not in sys.path:  # 解决VSCode没有ROOT的问题
    sys.path.append(str(ROOT))
ROOT = ROOT.relative_to(Path.cwd())

from utils.general import (
    IMAGE_TYPE, RECORDER, TranslationDict,
    get_logger, colorstr, listdir, second_confirm, fix_label_format,
    split_list_equally, calc_cost_time, dict2table, statistics
)


class Creator():
    def __init__(self, dataset: Path):
        self.dataset = Path(dataset)


class VOC2012Creator(Creator):
    def __init__(self, dataset: Path):
        # 根据数据集样式，创建所需要的文件夹的对象
        self.dataset = Path(dataset)
        self.VOC2012 = self.dataset.joinpath('VOC2012')
        self.Annotations = self.VOC2012.joinpath('Annotations')
        self.JPEGImages = self.VOC2012.joinpath('JPEGImages')
        self.ImageSets = self.VOC2012.joinpath('ImageSets')
        self.Main = self.ImageSets.joinpath('Main')

    
    def mkdir_better(self, d: Path, parents=False, exist_ok=False):
        if d.exists():
            if d.iterdir():
                LOGGER.warning(f"⚠️ The directory {colorstr('underline', str(d))} has existed, and it is not empty!")
        else:
            d.mkdir(parents=parents, exist_ok=exist_ok)

    
    def mkdir(self):
        self.mkdir_better(self.VOC2012)
        self.mkdir_better(self.Annotations)
        self.mkdir_better(self.JPEGImages)
        self.mkdir_better(self.ImageSets)
        self.mkdir_better(self.Main)

    def move(self, image: Path, label: Path, stage: int):
        # stage is useless in this creator
        image_dst = self.JPEGImages.joinpath(image.name)
        label_dst = self.Annotations.joinpath(label.name)

        image.rename(image_dst)
        label.rename(label_dst)
        

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, default="Datasets/raw_data/images", help="图片路径")
    parser.add_argument("--label-path", type=str, default="Datasets/raw_data/jsons", help="标签路径")
    parser.add_argument("--label-format", type=str, default=".json", help="标签的后缀")
    parser.add_argument("--dataset-path", "--target-path", type=str, default="Datasets/voc2012-Le0v1n", help="数据集保存路径")
    parser.add_argument("--dataset-style", "--target-style", '--style', type=str, default="voc2012", help="生成的数据集样式，可选：'voc2007', 'voc2012', 'coco'")
    parser.add_argument("--split-ratio", type=float, nargs=3, default=[0.8, 0.2, 0.0], help="数据集划分比例，务必传入3个float")
    parser.add_argument("--num-threading", type=int, default=8, help="使用的线程数，不使用多线程则设置为1")
    
    return parser.parse_known_args()[0] if known else parser.parse_args()


def process(args: argparse, images: list, stage: int, creator: Creator) -> None:
    assert isinstance(stage, int) and 0 <= stage <= 2, colorstr('red', 'bold', f"❌ The process function need correct stage instead of {stage}!")
    for image in images:  # image: PosixPath
        RECORDER["touch"] += 1
        image = Path(image)  # 为了方便IDE给出代码提示

        # 更新进度条显示信息
        prefix = '[Train]' if stage == 0 else '[Val]' if stage == 1 else '[Test]'
        prefix = colorstr(prefix)
        pbar.set_description(f"{prefix} Processing {colorstr(image.name):<30s}")

        # 找到对应的标签文件
        label = label_path.joinpath(image.stem + args.label_format)

        # 判断图片和标签是否存在
        if not image.exists():
            LOGGER.error(f"❌ The image {colorstr('underline', str(image))} don't exist!")
        if not label.exists():
            LOGGER.error(f"❌ The image {colorstr('underline', str(image))} corresponding label {colorstr('underline', str(label))} don't exist!")

        # 开始移动
        creator.move(image, label, stage)

        RECORDER["found"] += 1
        pbar.update()


def fix_dataset_style(style: str) -> str:
    if isinstance(style, str):
        style = style.lower()

        if style in ('voc2007', ):
            return 'voc2007'
        elif style in ('voc', 'voc2012', ):
            return 'voc2012'
        elif style in ('coco', 'coco128', 'mscoco', 'ms coco'):
            return 'coco'
        else:
            raise NotImplementedError(
                f"❌ The current dataset-style only supports {colorstr('voc2007, voc2012, coco')}, "
                f"and does not support {colorstr(style)}!"
            )
    else:
        raise TypeError(f"❌ The type of dataset-style should be {colorstr('str')} instead of {colorstr(type(style))}!")
    

def split_images(images: list, split_ratio: list, seed=42) -> tuple:
    # 对list进行shuffle
    random.seed(seed)
    random.shuffle(images)

    # 计算具体的数值
    num_samples = len(images)
    num_train = math.ceil(split_ratio[0] * num_samples)
    num_val = math.floor(split_ratio[1] * num_samples)
    num_test = round(split_ratio[2] * num_samples)

    num_rest = num_samples - num_train - num_val - num_test

    if num_rest > 0:
        num_train += num_rest

    return images[0: num_train], images[num_train: num_train + num_val], images[num_train + num_val: ]


if __name__ == "__main__":
    t1 = time.time()
    LOGGER = get_logger(FILE)  # global
    
    # 解析参数
    args = parse_opt(known=False)  # 如果发现不认识的参数则报错

    # 检查并修正标签后缀和数据集样式
    args.label_format = fix_label_format(args.label_format)
    args.dataset_style = fix_dataset_style(args.dataset_style)

    # 清空字典
    RECORDER.clear()

    # 记录
    RECORDER['image path'] = args.image_path
    RECORDER['label path'] = args.label_path
    RECORDER['label suffix'] = args.label_format
    RECORDER['dataset style'] = args.dataset_style
    
    # 读取所有的图片和标签
    total_images = listdir(args.image_path, extension=IMAGE_TYPE)
    total_labels = listdir(args.label_path, extension=args.label_format)
    RECORDER['images'] = len(total_images)
    RECORDER['labels'] = len(total_labels)

    # 确保图片和标签的数量一致
    if round(sum(args.split_ratio), 10) != 1.0:  # 需要使用round来控制，否则会出现0.999999999的情况
        LOGGER.error(colorstr('red', 'bold', f"⚠️ The split-ratio ({args.split_ratio}) is illegal"))
        exit(1)
    if RECORDER['images'] != RECORDER['labels']:
        LOGGER.error(colorstr('red', 'bold', f"⚠️ The number of images ({RECORDER['images']}) and labels ({RECORDER['labels']}) does not match!"))
        exit(1)
    
    # 划分数据集
    train_image_list, val_image_list, test_image_list = split_images(total_images, args.split_ratio)

    # 记录信息
    RECORDER['num train'] = len(train_image_list)
    RECORDER['num val'] = len(val_image_list)
    RECORDER['num test'] = len(test_image_list)
    if RECORDER['num train'] + RECORDER['num val'] + RECORDER['num test'] != RECORDER['images']:
        LOGGER.error(colorstr('red', 'bold', f"⚠️ The dataset split occurs some issue, please check corresponding code!"))
        exit(1)

    # 根据线程数，得到每个线程需要处理的训练集、验证集、测试集图片
    train_total_image_lists = split_list_equally(train_image_list, args.num_threading)
    val_total_image_lists = split_list_equally(val_image_list, args.num_threading)
    test_total_image_lists = split_list_equally(test_image_list, args.num_threading)

    # 记录线程相关
    RECORDER['threadings'] = args.num_threading
    RECORDER['data num of every threading'] = [len(train_total_image_lists[0]), len(val_total_image_lists[0]), len(test_total_image_lists[0])]
    RECORDER['script'] = str(FILE.name)
    
    # 输出开始执行脚本前的统计信息
    LOGGER.info(dict2table(RECORDER, align='l', replace_keys=TranslationDict))

    # 2FA
    second_confirm(script=FILE, LOGGER=LOGGER)

    # 记录
    RECORDER['touch'] = 0
    RECORDER['found'] = 0
    
    # 创建Path对象
    dataset_path = Path(args.dataset_path)
    label_path = Path(args.label_path)
    
    # 创建标签文件夹
    if not dataset_path.parent.exists():
        second_confirm(msg=colorstr(f"⚠️ The parent directory {colorstr('underline', str(dataset_path.parent))} don't exist, do you want to create it?"), script=FILE, LOGGER=LOGGER)
        dataset_path.parent.mkdir(parents=True, exist_ok=False)
    dataset_path.mkdir(parents=False, exist_ok=True)

    # 根据args.dataset_style使用不同的数据集创建器
    if args.dataset_style.lower() == 'voc2007':
        ...
    elif args.dataset_style.lower() == 'voc2012':
        creator = VOC2012Creator(dataset_path)
    elif args.dataset_style.lower() == 'coco':
        ...

    # 根据数据集样式，创建所需要的空文件夹
    creator.mkdir()
    
    for stage in range(3):
        # stage: 0: train, 1: val, 2: test
        threads = []  # 保存线程的list
        pbar = tqdm(
            total=RECORDER['num train'] if stage == 0 else RECORDER['num val'] if stage == 1 else RECORDER['num test'], 
            dynamic_ncols=True
        )  # for every image file
        for images in train_total_image_lists if stage == 0 else val_total_image_lists if stage == 1 else test_total_image_lists:
            t = threading.Thread(
                target=process, 
                args=(
                    args, 
                    images,
                    stage,
                    creator
                )
            )
            threads.append(t)
            t.start()

        # 等待所有线程都执行完毕
        for t in threads:
            t.join()

        # 所有进程结束后再关闭进度条
        pbar.close()

    # 统计正样本情况
    RECORDER = statistics(RECORDER)
    
    # 再次输出统计信息
    LOGGER.info(dict2table(RECORDER, align='l', replace_keys=TranslationDict))
    
    if RECORDER["found"]  == RECORDER["images"]:
        LOGGER.info(colorstr('green', 'bold', '✅ All negative labels has created correctly!'))
    else:
        LOGGER.warning(colorstr('red', 'bold', "⚠️ Some question have occurred, please check dataset!"))

    LOGGER.info(f"⏳ The cost time of {str(FILE.name)} is {colorstr(calc_cost_time(t1, time.time()))}")
    LOGGER.info(
        f"👀 The detailed information has been saved to {colorstr(LOGGER.handlers[0].baseFilename)}. \n"
        f"    This script is formatted with {colorstr('ANSI')} color codes, so it is recommended to {colorstr('use a terminal or a compatible tool')} "
        f"that supports color display for viewing."
    )
