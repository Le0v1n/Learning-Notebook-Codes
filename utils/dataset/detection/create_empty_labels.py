import sys
import argparse
import threading
from pathlib import Path
from PIL import Image
from prettytable import PrettyTable
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
    TXTWriter, XMLWriter, JsonWriter, split_list_equally, calc_cost_time, dict2table, statistics
)
        

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, default="Datasets/coco128/train/images", help="图片/负样本路径")
    parser.add_argument("--target-path", type=str, default="Datasets/coco128/train/negative_labels", help="空标签保存路径")
    parser.add_argument("--target-format", type=str, default=".txt", help="空标签格式，可选：'.txt', '.json', '.xml'")
    parser.add_argument("--override", action='store_true', default=False, help="如果对应的空标签文件存在，是否覆盖它")
    parser.add_argument("--num-threading", type=int, default=8, help="使用的线程数，不使用多线程则设置为1")
    
    return parser.parse_known_args()[0] if known else parser.parse_args()


def process(args: argparse, images: list) -> None:
    for image in images:  # image: PosixPath
        image = Path(image)  # 为了方便IDE给出代码提示

        # 更新进度条显示信息
        pbar.set_description(f"Processing {colorstr(image.name):<30s}")

        RECORDER["touch"] += 1
        
        # 读取图片
        im = Image.open(image)

        # 验证图片是否破损
        if not verify_image(image):  # 验证图片是否破损
            pbar.clear()
            LOGGER.error(f"❌ [Corrupt image] Found corrupt image! -> {str(image)}")
            RECORDER["corrupt"] += 1
            pbar.update()
            continue
        
        # 获取图片尺寸和通道数
        img_width, img_height = exif_size(im)
        img_channel = 3 if im.mode == "RGB" else 1 if im.mode == "L" else 4 if im.mode == "RGBA" else "Unknown"
        
        # 如果target文件存在
        target = target_dir.joinpath(image.stem + args.target_format)
        if target.exists() and target.read_text():  # 如果文件存在且文件内容不为空
            if args.override:  # 覆盖掉之前的内容
                pbar.clear()
                LOGGER.warning(f"⚠️ [Override] The target file has existed, but its content will be overrode! -> {str(target)}")
            else:
                pbar.clear()
                LOGGER.info(f"[Skip] The target file has existed, and it will not be overrode. -> {str(target)}")
                RECORDER['skip'] += 1
                pbar.update()
                continue
        
        if args.target_format == '.txt':
            writer = TXTWriter()
        elif args.target_format == '.xml':
            writer = XMLWriter(image, img_width, img_height, img_channel)
        elif args.target_format == '.json':
            writer = JsonWriter(image, img_width, img_height)

        # 保存文件
        writer.save(target)

        RECORDER["found"] += 1
        pbar.update()
    

if __name__ == "__main__":
    t1 = time.time()
    LOGGER = get_logger(FILE)  # global
    
    # 解析参数
    args = parse_opt(known=False)  # 如果发现不认识的参数则报错

    # 检查并修正标签后缀
    args.target_format = fix_label_format(args.target_format)

    # 清空字典
    RECORDER.clear()

    # 记录
    RECORDER['image path'] = args.image_path
    RECORDER['target path'] = args.target_path
    RECORDER['target format'] = args.target_format
    
    # 读取所有的图片和标签
    total_images = listdir(args.image_path, extension=IMAGE_TYPE)
    RECORDER['images'] = len(total_images)

    # 根据线程数，得到每个线程需要处理的图片list
    total_image_lists = split_list_equally(total_images, args.num_threading)

    # 记录线程相关
    RECORDER['threadings'] = args.num_threading
    RECORDER['data num of every threading'] = len(total_image_lists[0])
    RECORDER['script'] = str(FILE.name)
    
    # 输出开始执行脚本前的统计信息
    LOGGER.info(dict2table(RECORDER, align='l', replace_keys=TranslationDict))
    
    # 2FA
    second_confirm(script=FILE)

    # 记录
    RECORDER['touch'] = 0
    RECORDER['corrupt'] = 0
    RECORDER['skip'] = 0
    RECORDER['found'] = 0
    
    # 创建Path对象
    target_dir = Path(args.target_path)
    
    # 创建标签文件夹
    target_dir.mkdir(exist_ok=True)
    
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

    # 统计正样本情况
    RECORDER = statistics(RECORDER)
    
    # 再次输出统计信息
    LOGGER.info(dict2table(RECORDER, align='l', replace_keys=TranslationDict))
    
    if RECORDER["found"] + RECORDER['skip'] == RECORDER["images"]:
        LOGGER.info(colorstr('green', 'bold', '✅ All negative labels has created correctly!'))
    else:
        LOGGER.warning(colorstr('red', 'bold', "⚠️ Some question have occurred, please check dataset!"))

    if RECORDER['skip'] == RECORDER['images']:
        LOGGER.warning(f"⚠️ All target file have been skipped, please check dataset!")

    LOGGER.info(f"⏳ The cost time of {str(FILE.name)} is {colorstr(calc_cost_time(t1, time.time()))}")
    LOGGER.info(
        f"👀 The detailed information has been saved to {colorstr(LOGGER.handlers[0].baseFilename)}. \n"
        f"    This script is formatted with {colorstr('ANSI')} color codes, so it is recommended to {colorstr('use a terminal or a compatible tool')} "
        f"that supports color display for viewing."
    )
