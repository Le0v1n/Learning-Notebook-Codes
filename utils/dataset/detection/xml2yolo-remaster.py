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
    IMAGE_TYPE, RECORDER, 
    get_logger, colorstr, listdir, second_confirm, verify_image, exif_size, read_xml, fix_illegal_coordinates, 
    fix_reverse_coordinates, xyxy2xywh, split_list_equally, calc_cost_time, check_dataset, dict2table, 
    reverse_dict, statistics)
        

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, default="Datasets/coco128/train/images", help="图片路径")
    parser.add_argument("--label-path", type=str, default="Datasets/coco128/train/annotations-xml", help="xml标签路径")
    parser.add_argument("--target-path", type=str, default="Datasets/coco128/train/labels", help="yolo标签路径保存路径")
    parser.add_argument("--classes", type=str, nargs='+', default=['cat', 'dog'], help="数据集标签")
    parser.add_argument("--override", action='store_true', default=False, help="如果对应的.txt文件存在，是否覆盖它")
    parser.add_argument("--num-threading", type=int, default=4, help="使用的线程数，不使用多线程则设置为1")
    
    return parser.parse_known_args()[0] if known else parser.parse_args()


def process(args: argparse, images: list) -> None:
    for image in images:  # image: PosixPath
        image = Path(image)  # 为了方便IDE给出代码提示

        # 更新进度条显示信息
        pbar.set_description(f"Processing {colorstr(image.name):<30s}")

        RECORDER["touch"] += 1
        
        # 读取图片尺寸
        im = Image.open(image)

        # 验证图片是否破损
        if not verify_image(image):  # 验证图片是否破损
            pbar.clear()
            LOGGER.error(f"❌ [Corrupt image] Found corrupt image! -> {str(image)}")
            RECORDER["corrupt"] += 1
            pbar.update()
            continue
        
        # 获取图片尺寸
        img_width, img_height = exif_size(im)
        
        # 确定label位置
        label = label_dir.joinpath(image.stem + '.xml')
        
        # 判断label是否存在：不存在 -> 负样本
        if not label.exists():
            pbar.clear()
            LOGGER.info(f"⚠️ [Negative sample] {str(image)}")
            RECORDER["missing"] += 1
            pbar.update()
            continue
            
        # 读取label信息并获取"object"信息
        label_data = read_xml(label)
        objects = label_data.get("object", None)

        # 如果没有object -> 定义为负样本
        if not objects:
            RECORDER["background"] += 1
            pbar.update()
            continue
        
        # 如果target文件存在
        target = target_dir.joinpath(image.stem + '.txt')
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
        
        # 处理objects文件
        for index, object in enumerate(objects):
            # 检查：坐标点的个数是否为4
            num_pts = len(object["bndbox"])
            if num_pts != 4:
                pbar.clear()
                LOGGER.error(f"❌ [Illegal points] The No.{index} object has illegal points({num_pts} != 4)! -> {str(label)}")
                RECORDER["illegal_pts"] += 1
                pbar.update()
                continue
            
            # 获取每个object的box信息
            x1 = float(object["bndbox"]["xmin"])
            y1 = float(object["bndbox"]["ymin"])
            x2 = float(object["bndbox"]["xmax"])
            y2 = float(object["bndbox"]["ymax"])
            
            # 检查：修复不合规的坐标：负数和越界
            x1, y1, x2, y2, msg = fix_illegal_coordinates(
                x1, y1, x2, y2, img_width, img_height
            )
            if msg:
                msg = [f"[{i}] {content}" for i, content in enumerate(msg)]
                msg = ", ".join(msg)
                pbar.clear()
                LOGGER.warning(f"⚠️ [Out of boundary] The No.{index} object has illegal coordinates: {msg}! -> {str(label)}")
                RECORDER["out_of_boundary"] += 1
            
            # 检查：修复相反的坐标：x2y2x1y1 -> x1y1x2y2
            x1, y1, x2, y2, msg = fix_reverse_coordinates(x1, y1, x2, y2)
            if msg:
                msg = [f"[{i}] {content}" for i, content in enumerate(msg)]
                msg = ", ".join(msg)
                pbar.clear()
                LOGGER.warning(f"⚠️ [Reversed coordinates] The No.{index} object of has illegal coordinates: {msg}! -> {str(label)}")
                RECORDER["reversed"] += 1
            
            # 获取对应的类别并转换为索引
            class_name = object["name"]
            try:
                class_index = classes_dict[class_name]
            except:
                pbar.clear()
                LOGGER.error(f"❌ [Unknown class name] The class {class_name} don't exist in {classes_dict}! -> {str(label)}")
                exit(f"❌ {class_name} of {str(label)} don't exist in {classes_dict}!")

            # xyxy2xywh
            x, y, w, h = xyxy2xywh(x1, y1, x2, y2)

            # 绝对坐标转相对坐标，保存6位小数
            x = round(x / img_width, 6)
            y = round(y / img_height, 6)
            w = round(w / img_width, 6)
            h = round(h / img_height, 6)
            
            # 要输入txt文本的内容
            info = [str(i) for i in [class_index, x, y, w, h]]  # c, x, y, w, h

            # 写入txt
            with target.open('a') as f:  # ☣️ 注意这里应该用'a'而非'w'，否则每次文件都会被清空的！
                if index == 0:
                    f.write(" ".join(info))
                else:
                    f.write("\n" + " ".join(info))

            RECORDER['objects'] += 1  # 记录对象+1

        RECORDER["found"] += 1
        pbar.update()
    

if __name__ == "__main__":
    t1 = time.time()
    LOGGER = get_logger(FILE)  # global
    
    # 解析参数
    args = parse_opt(known=False)  # 如果发现不认识的参数则报错
    
    # 读取所有的图片和标签
    total_images = listdir(args.image_path, extension=IMAGE_TYPE)
    total_labels = listdir(args.label_path, extension='.xml')
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
    replace_keys = {
        'found': '正样本(有标签)数量',
        'missing': '负样本(没有标签)数量',
        'corrupt': '破损的图片数量',
        'illegal_pts': '坐标点不足4的数量',
        'out_of_boundary': '坐标越界的数量',
        'reversed': '坐标点反了的数量',
        'background': '负样本(有空标签)的数量',
        'touch': '程序touch过的数量',
        'skip': '跳过(目标标签存在)的数量',
        'images': '图片的数量',
        'labels': 'xml标签的数量',
        'nc': '类别数',
        'threadings': '使用的线程数',
        'data num of every threading': '单个线程的并发量',
        'script': '脚本名称',
        'objects': '对象总数',
        'average objects for all': '平均每张图片的对象数',
        'average objects for positives': '平均每张正样本图片的对象数',
        'positive ratio': '正样本比例',
        'negative ratio': '负样本比例',
    }
    LOGGER.info(dict2table(RECORDER, align='l', replace_keys=replace_keys))
    
    # 根据图片和标签数量发出对应的告警
    check_dataset(num_images=RECORDER['images'], num_labels=RECORDER['labels'])
    
    # 2FA
    second_confirm(script=FILE)
    
    # 创建Path对象
    label_dir = Path(args.label_path)
    target_dir = Path(args.target_path)
    
    # 创建标签文件夹
    target_dir.mkdir(exist_ok=True)
    
    threads = []  # 保存线程的list
    pbar = tqdm(total_images, dynamic_ncols=True)  # for every image file
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
    LOGGER.info(dict2table(RECORDER, align='l', replace_keys=replace_keys))
    
    if RECORDER["found"] + RECORDER["background"] + RECORDER['skip'] + RECORDER['missing'] == RECORDER["images"]:
        LOGGER.info(colorstr('green', 'bold', '✅ All conversion has done correctly!'))
        if RECORDER['missing'] != 0:
            LOGGER.warning(colorstr('yellow', 'bold', f"⚠️ There are {RECORDER['missing']} images without label, "
                                    f"and they have be regarded as negative samples!"))
    else:
        LOGGER.warning(colorstr('red', 'bold', "⚠️ Some question have occurred, please check dataset!"))

    if RECORDER['skip'] == RECORDER['images']:
        LOGGER.warning(f"⚠️ All target file have been skipped, please check dataset!")

    LOGGER.info(f"⏳ The cost time of {str(FILE.name)} is {colorstr(calc_cost_time(t1, time.time()))}")
    LOGGER.info(
        f"👀 The detailed information has been saved to {colorstr(LOGGER.handlers[0].baseFilename)}. \n"
        f"    This file is formatted with ANSI color codes, so it is recommended to use a terminal or a compatible "
        f"tool that supports color display for viewing."
    )
