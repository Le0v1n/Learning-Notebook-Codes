import sys
import argparse
import threading
from pathlib import Path
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
    get_logger, colorstr, listdir, second_confirm, fix_label_format, verify_image, exif_size,  
    LabelVerifier, split_list_equally, calc_cost_time, dict2table, reverse_dict
)

__doc__ = """
- 脚本功能：
    1. 检查图片和标签，挑选出所有的负样本（图片+标签）
    2. 挑选出没有对应图片的标签

- 转换前目录结构：
    Datasets/coco128
    ├── images  # 所有的图片
    └── labels  # 所有的标签（支持.txt、.xml、.json）

- 转换后目录结构：
    Datasets/coco128
    ├── images
    ├── labels
    └── negative_samples  # 负样本（图片+标签）
        ├── corrupt      # 图片破损
        │   ├── images
        │   └── labels
        ├── background   # 背景
        │   ├── images
        │   └── labels
        ├── label_issue  # 标签内容有问题
        │   ├── images
        │   └── labels
        └── redundant    # 冗余的标签文件
            └── labels

- 主要功能：
    - 功能1：检查图片和标签
        - yolo(txt)：
            1. 检查图片是否破损，如破损，则图片和标签（如果存在）均被判定为corrupt
            2. 检查图片对应的标签文件是否存在，不存在则判定为background
            3. 检查标签文件是否有内容，如标签文件为空，则图片和标签均被判定为background
            4. 检查坐标点是否完整，如不完整，则图片和标签均被判定为background
            5. 检查类别是否越界，如越界，则图片和标签均被判定为background

        - xml：
            1. 检查图片是否破损，如破损，则图片和标签（如果存在）均被判定为corrupt
            2. 检查图片对应的标签文件是否存在，不存在则判定为background
            3. 检查标签文件是否有内容，如标签文件为空，则图片和标签均被判定为background
            4. 检查坐标点是否完整，如不完整，则图片和标签均被判定为background
            5. 检查类别是否越界，如越界，则图片和标签均被判定为background
            6. 检查标签中的文件名是否与实际图片一致（实际图片名称在标签文件名中即可），如不满足，图片和标签均被判定为label issue
            7. 检查标签中的图片尺寸（宽度、高度、通道数）是否与实际图片一致，如不满足，图片和标签均被判定为label issue

        - json：
            1. 检查图片是否破损，如破损，则图片和标签（如果存在）均被判定为corrupt
            2. 检查图片对应的标签文件是否存在，不存在则判定为background
            3. 检查标签文件是否有内容，如标签文件为空，则图片和标签均被判定为background
            4. 检查坐标点是否完整，如不完整，则图片和标签均被判定为background
            5. 检查类别是否越界，如越界，则图片和标签均被判定为background
            6. 检查标签中的文件名是否与实际图片一致（实际图片名称在标签文件名中即可），如不满足，图片和标签均被判定为label issue
            7. 检查标签中的图片尺寸（宽度、高度）是否与实际图片一致，如不满足，图片和标签均被判定为label issue
            8. 检查标签imageData是否为空，如不满足，图片和标签均被判定为label issue
               💡 不为空算违规，可自行在utils/general.py文件的LabelVerifier的verify_metadata方法中取消该判定

    - 功能2：检查是否有冗余的标签（检查是否有没有对应图片的标签）
        在程序结束后会自动检查--label-path中的标签是否都有对应的图片文件，如果发现冗余的标签，则该标签被判定为redundant

- 其他说明：
    1. 该脚本不会修改文件内容，只会对文件进行移动。
    2. 移动的destination根据违规判断而改变，具体为：
        - corrupt：图片被判定为破损
            图片：--target-path/corrupt/images
            标签：--target-path/corrupt/labels
        - bakcground：图片被判断为背景（没有对应的标签文件）
            图片：--target-path/bakcground/images
        - label issue：标签有问题
            图片：--target-path/label_issue/images
            标签：--target-path/label_issue/labels
        - redundant：有冗余的标签文件
            图片：--target-path/redundant/images
            标签：--target-path/redundant/labels
"""
        

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, default="Datasets/coco128/train/images", help="图片路径")
    parser.add_argument("--label-path", type=str, default="Datasets/coco128/train/labels", help="标签路径")
    parser.add_argument("--label-format", type=str, default=".txt", help="标签格式，可选：'.txt', '.json', '.xml'")
    parser.add_argument("--classes", type=str, nargs='+', default=['cat', 'dog'], help="数据集标签")
    parser.add_argument("--target-path", type=str, default="Datasets/coco128/train/negative_samples", help="负样本保存路径")
    parser.add_argument("--num-threading", type=int, default=8, help="使用的线程数，不使用多线程则设置为1")
    
    return parser.parse_known_args()[0] if known else parser.parse_args()


def move(src_image: Path, src_label: Path, dst_dir: Path, reason: str = 'background') -> tuple:
    assert dst_dir, f"❌  The move function must have dst_dir!"

    if reason:
        dst_dir = dst_dir.joinpath(reason)
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_image = None
    dst_label = None

    # 移动图片
    if src_image:
        dst_dir_image = dst_dir.joinpath('images')
        dst_dir_image.mkdir(exist_ok=True)
        dst_image = dst_dir_image.joinpath(src_image.name)
        src_image.rename(dst_image) if src_image.exists() else ...

    # 移动图片对应的标签
    if src_label:
        dst_dir_label = dst_dir.joinpath('labels')
        dst_dir_label.mkdir(exist_ok=True)
        dst_label = dst_dir_label.joinpath(src_label.name)
        src_label.rename(dst_label) if src_label.exists() else ...

    return dst_image if dst_image else src_image, dst_label if dst_label else src_label
        

def process(args: argparse, images: list) -> None:
    for image in images:  # image: PosixPath
        image = Path(image)  # 为了方便IDE给出代码提示

        # 更新进度条显示信息
        pbar.set_description(f"Processing {colorstr(image.name):<30s}")

        RECORDER["touch"] += 1

        # 确定label位置
        label = label_dir.joinpath(image.stem + args.label_format)

        # 验证图片是否正确
        if not verify_image(image):
            # 移动负样本
            image, label = move(
                src_image=image,
                src_label=label,
                dst_dir=target_dir,
                reason='corrupt'
            )

            pbar.clear()
            LOGGER.error(f"❌ [Corrupt image] Found corrupt image! -> {str(image)}")
            RECORDER["corrupt"] += 1
            pbar.update()

            continue

        # 获取图片尺寸和通道数
        im = Image.open(image)
        img_width, img_height = exif_size(im)
        img_channel = 3 if im.mode == "RGB" else 1 if im.mode == "L" else 4 if im.mode == "RGBA" else "Unknown"

        # 创建标签检查器，进一步检查标签是否正确
        label_verifier = LabelVerifier(image, label, classes_dict, img_width, img_height, img_channel)

        # 先检查对应的标签文件是否存在
        if not label_verifier.label_exists():  # 如果对应的标签文件不存在 -> 负样本
            # 移动负样本
            image, label = move(
                src_image=image,
                src_label=label,
                dst_dir=target_dir,
                reason='background'
            )

            RECORDER['label not exist'] += 1
            pbar.clear()
            LOGGER.info(f"⚠️ [label not exist] The corresponding label don't exist! -> {str(image)}")
            pbar.update()
            continue

        # 标签文件存在，进一步检查标签是否正确
        verify_label_message = label_verifier.start_and_receive_results()
        if verify_label_message:
            # 移动负样本
            image, label = move(
                src_image=image,
                src_label=label,
                dst_dir=target_dir,
                reason='label_issue'
            )

            pbar.clear()
            LOGGER.error(f"❌ [label issue] Found some issue of label: {verify_label_message} -> {str(label)}")
            RECORDER["label issue"] += 1
            pbar.update()
        pbar.update()


def find_redundant_label_and_move(labels: list, image_dir: Path, target_dir: Path):
    for label in labels:
        label = Path(label)
        pbar.set_description(f"Processing {colorstr(label.name):<30s}")

        # 创建对应图片的Path对象
        exists = 0
        for ext in IMAGE_TYPE:
            image = Path(image_dir).joinpath(label.stem + ext)
            if image.exists():
                exists += 1
        
        if exists <= 0:  # 对应的图片不存在
            _, label = move(
                src_image=None,
                src_label=label,
                dst_dir=target_dir,
                reason='redundant'
            )
            RECORDER['redundant'] += 1
            pbar.clear()
            LOGGER.info(f"⚠️ [redundant label] Found a redundant label -> {str(label)}")
        pbar.update()


if __name__ == "__main__":
    t1 = time.time()
    LOGGER = get_logger(FILE)  # global
    
    # 解析参数
    args = parse_opt(known=False)  # 如果发现不认识的参数则报错

    # 检查并修正标签后缀
    args.label_format = fix_label_format(args.label_format)

    # 清空字典
    RECORDER.clear()

    # 记录
    RECORDER['image path'] = args.image_path
    RECORDER['label path'] = args.label_path
    RECORDER['target path'] = args.target_path
    RECORDER['label format'] = args.label_format
    
    # 读取所有的图片和标签
    total_images = listdir(args.image_path, extension=IMAGE_TYPE)
    total_labels = listdir(args.label_path, extension=args.label_format)
    RECORDER['images'] = len(total_images)
    RECORDER['labels'] = len(total_labels)

    # 创建类别字典
    classes_dict = {i: cla for i, cla in enumerate(args.classes)}  # int: str, e.g. {0: 'cat', 1: 'dog'}
    RECORDER['nc'] = len(args.classes)
    RECORDER['classes_dict'] = classes_dict

    # 💡 在改脚本中需要翻转一下字典
    classes_dict = reverse_dict(classes_dict)

    # 根据线程数，得到每个线程需要处理的图片list
    total_image_lists = split_list_equally(total_images, args.num_threading)

    # 记录线程相关
    RECORDER['threadings'] = args.num_threading
    RECORDER['data num of every threading'] = len(total_image_lists[0])
    RECORDER['script'] = str(FILE.name)
    
    # 输出开始执行脚本前的统计信息
    LOGGER.info(dict2table(RECORDER, align='l', replace_keys=TranslationDict))
    
    # 2FA
    second_confirm(script=FILE, LOGGER=LOGGER)

    # 记录
    RECORDER['touch'] = 0  # 处理过的图片数量
    RECORDER['corrupt'] = 0  # 破损图片的数量
    RECORDER['label not exist'] = 0  # 图片没有对应的标签文件
    RECORDER['label issue'] = 0  # 标签有问题的数量
    RECORDER['redundant'] = 0  # 标签没有对应图片的数量
    
    # 创建Path对象
    label_dir = Path(args.label_path)
    target_dir = Path(args.target_path)
    
    # 创建标签文件夹
    target_dir.mkdir(exist_ok=True)

    # ---------- 程序1：发现负样本 ----------
    threads = []  # 保存线程的list
    pbar = tqdm(total=RECORDER['images'], dynamic_ncols=True)  # for every image file
    for images in total_image_lists:
        t = threading.Thread(
            target=process, 
            args=(
                args, 
                images,
            )
        )
        threads.append(t)
        t.start()

    # 等待所有线程都执行完毕
    for t in threads:
        t.join()

    # 所有进程结束后再关闭进度条
    pbar.close()

    # ---------- 程序2：查找冗余的标签 ----------
    total_labels = listdir(args.label_path, extension=args.label_format)  # 再次读取所有的标签
    total_label_lists = split_list_equally(total_labels, args.num_threading)  # 根据线程数，得到每个线程需要处理的图片list
    image_dir = Path(args.image_path)

    threads = []  # 保存线程的list
    pbar = tqdm(total=len(total_labels), dynamic_ncols=True)  # for every image file
    for labels in total_label_lists:
        t = threading.Thread(
            target=find_redundant_label_and_move,  # 找到没有图片配对的标签
            args=(
                labels,
                image_dir,
                target_dir,
            )
        )
        threads.append(t)
        t.start()

    # 等待所有线程都执行完毕
    for t in threads:
        t.join()

    # 所有进程结束后再关闭进度条
    pbar.close()
    
    # 再次输出统计信息
    LOGGER.info(dict2table(RECORDER, align='l', replace_keys=TranslationDict))
    
    if RECORDER['touch'] == RECORDER['images'] and RECORDER["corrupt"] + RECORDER['label issue'] == 0:
        LOGGER.info(colorstr('green', 'bold', '✅ All negative labels has created correctly!'))
    else:
        LOGGER.warning(colorstr('red', 'bold', "⚠️ Some question have occurred, please check dataset and fix them!"))

    if RECORDER['redundant'] != 0:
        LOGGER.info(colorstr('bright_yellow', 'bold', f"⚠️ Found {RECORDER['redundant']} redundant label!"))

    LOGGER.info(f"⏳ The cost time of {str(FILE.name)} is {colorstr(calc_cost_time(t1, time.time()))}")
    LOGGER.info(
        f"👀 The detailed information has been saved to {colorstr(LOGGER.handlers[0].baseFilename)}. \n"
        f"    This script is formatted with {colorstr('ANSI')} color codes, so it is recommended to {colorstr('use a terminal or a compatible tool')} "
        f"that supports color display for viewing."
    )
