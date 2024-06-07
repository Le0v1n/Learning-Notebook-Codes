import sys
import argparse
import threading
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from prettytable import PrettyTable
import time


ROOT = Path.cwd().resolve()
FILE = Path(__file__).resolve()  # 当前脚本的绝对路径
if str(ROOT) not in sys.path:  # 解决VSCode没有ROOT的问题
    sys.path.append(str(ROOT))
ROOT = ROOT.relative_to(Path.cwd())

from utils.general import (LOGGER, colorstr, listdir, second_confirm, 
                           verify_image, read_xml, fix_illegal_coordinates, 
                           fix_reverse_coordinates, xyxy2xywh,
                           split_list_equally, calc_cost_time,
                           dataset_number_comparison)
        

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, default="Datasets/coco128/train/images", help="图片路径")
    parser.add_argument("--label-path", type=str, default="Datasets/coco128/train/annotations-xml", help="xml标签路径")
    parser.add_argument("--target-path", type=str, default="Datasets/coco128/train/labels", help="yolo标签路径保存路径")
    parser.add_argument("--classes", type=str, nargs='+', default=['cat', 'dog'], help="数据集标签")
    parser.add_argument("--image-format", type=str, nargs='+', default=['.png', '.jpg', '.jpeg', '.bmp', 'webp'], help="允许的图片格式")
    parser.add_argument("--override", action='store_true', default=False, help="如果对应的.txt文件存在，是否覆盖它")
    parser.add_argument("--num-threading", type=int, default=4, help="使用的线程数，不使用多线程则设置为1")
    
    return parser.parse_known_args()[0] if known else parser.parse_args()


def process(args: argparse, images: list) -> None:
    for image in images:  # image: PosixPath
        # update description of process bar 
        pbar.set_description(f"Processing {colorstr(image.name):<30s}")

        COUNTER["touch"] += 1
        
        # Get size of image
        img_width, img_height = Image.open(image).size
        if not verify_image(image):  # verify whether corrupts
            LOGGER.error(f"❌ [Corrupt image] Found corrupt image! -> {str(image)}")
            COUNTER["corrupt"] += 1
            pbar.update()
            continue
        
        # Open the corresponding image and get a dict
        xml = label_dir.joinpath(image.stem + '.xml')
        if not xml.exists():
            LOGGER.info(f"⚠️ [Negative sample] {str(image)}")
            COUNTER["missing"] += 1
            pbar.update()
            continue
        xml_data = read_xml(xml)
        
        # 定义如何处理.txt文件
        yolo = target_dir.joinpath(image.stem + '.txt')
        if yolo.exists() and yolo.read_text():  # 如果文件存在且文件内容不为空
            if args.override:  # override the previous content
                LOGGER.warning(f"⚠️ [Override] The target file has existed, but its content will be overrode! -> {str(yolo)}")
            else:
                LOGGER.info(f"[Skip] The target file has existed, and it will not be overrode. -> {str(yolo)}")
                COUNTER['skip'] += 1
                pbar.update()
                continue
        
        # 处理.txt文件
        with yolo.open('w') as f:
            objects = xml_data.get("object", None)
            if not objects:  # Negative samples
                COUNTER["background"] += 1
                pbar.update()
                continue
            
            # Positive samples
            for index, obj in enumerate(xml_data["object"]):
                # Check for the coordinates which the number should be 4).
                num_pts = len(obj["bndbox"])
                if num_pts != 4:
                    LOGGER.error(f"❌ [Incomplete points] The No.{index} object has incomplete points({num_pts} < 4)! -> {str(xml)}")
                    COUNTER["incomplete_pts"] += 1
                    continue
                
                # 获取每个object的box信息
                x1 = float(obj["bndbox"]["xmin"])
                y1 = float(obj["bndbox"]["ymin"])
                x2 = float(obj["bndbox"]["xmax"])
                y2 = float(obj["bndbox"]["ymax"])
                
                # 修复不合规的坐标：负数和越界
                x1, y1, x2, y2, msg = fix_illegal_coordinates(
                    x1, y1, x2, y2, img_width, img_height
                )
                if msg:
                    msg = [f"[{i}] {content}" for i, content in enumerate(msg)]
                    msg = ", ".join(msg)
                    LOGGER.warning(f"⚠️ [Out of boundary] The No.{index} object has illegal coordinates: {msg}! -> {str(xml)}")
                    COUNTER["out_of_boundary"] += 1
                
                # 修复相反的坐标：x2y2x1y1 -> x1y1x2y2
                x1, y1, x2, y2, msg = fix_reverse_coordinates(x1, y1, x2, y2)
                if msg:
                    msg = [f"[{i}] {content}" for i, content in enumerate(msg)]
                    msg = ", ".join(msg)
                    LOGGER.warning(f"⚠️ [Reversed coordinates] The No.{index} object of has illegal coordinates: {msg}! -> {str(xml)}")
                    COUNTER["reversed"] += 1
                
                # 获取对应的类别并转换为索引
                class_name = obj["name"]
                try:
                    class_index = classes_dict[class_name]
                except:
                    LOGGER.error(f"❌ [Unknown class name] The class {class_name} don't exist in {classes_dict}! -> {str(xml)}")
                    exit(f"❌ {class_name} of {str(xml)} don't exist in {classes_dict}!")

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
                if index == 0:
                    f.write(" ".join(info))
                else:
                    f.write("\n" + " ".join(info))
        COUNTER["found"] += 1
        pbar.update()
    pbar.close()
    
    
def dict2table(d: dict, field_names=['Key', 'Value'], align='l') -> PrettyTable:
    assert isinstance(d, dict), f"❌ print_dict() need a dict instead of {type(d)}!"

    ptab = PrettyTable(field_names)
    ptab.align = align
    
    
    for k, v in d.items():
        # process 'classes_dict'
        if isinstance(k, str) and k.lower() == 'classes_dict':
            for i, cla in v.items():
                ptab.add_row([f"    {i}", cla])
        else:
            ptab.add_row([k, v])
    
    return ptab
    

if __name__ == "__main__":
    t1 = time.time()
    
    # 解析参数
    args = parse_opt(known=False)  # 如果发现不认识的参数则报错

    # 创建一个计数器字典 -> global
    COUNTER = {
        'found': 0,  # 完成转换的标签
        'missing': 0,  # 缺少标签的图片数量
        'corrupt': 0,  # 图片破损的数量
        'incomplete_pts': 0,  # 标签点的数量不为4
        'out_of_boundary': 0,  # 坐标点越界
        'reversed': 0,  # 坐标点反了
        'background': 0,  # 负样本图片的数量
        'touch': 0,  # 触摸过的图片数量
        'skip': 0,  # 目标文件存在，跳过的数量
    }
    
    # 读取所有的图片和标签
    total_images = listdir(args.image_path, extension=args.image_format)
    total_labels = listdir(args.label_path, extension='.xml')
    COUNTER['images'] = len(total_images)
    COUNTER['labels'] = len(total_labels)
    
    # 创建类别字典
    classes_dict = {cla: i for i, cla in enumerate(args.classes)}  # str: int, e.g. {'cat': 0, 'dog': 1}
    COUNTER['nc'] = len(args.classes)
    COUNTER['classes_dict', classes_dict]

    # 根据线程数，得到每个线程需要处理的图片list
    total_image_lists = split_list_equally(total_images, args.num_threading)
    
    LOGGER.info(dict2table(COUNTER))
    exit()

    ptab = PrettyTable(['参数', '详情'])
    ptab.align = 'l'
    ptab.add_row(['图片路径', args.image_path])
    ptab.add_row(['图片数量', COUNTER['images']])
    ptab.add_row(['XML路径', args.label_path])
    ptab.add_row(['XML数量', COUNTER['labels']])
    ptab.add_row(['类别数', COUNTER['nc']])
    ptab.add_row(['类别', ''])
    for i, cla in classes_dict.items():
        ptab.add_row([f"    {i}", cla])
    ptab.add_row(['线程数', args.num_threading])
    ptab.add_row(['并发量/线程', len(total_image_lists[0])])
    LOGGER.info(ptab)
    
    # 根据图片和标签数量发出对应的告警
    dataset_number_comparison(num_images=COUNTER['images'], num_labels=COUNTER['labels'])
    
    # 2FA
    second_confirm()
    
    # 创建Path对象
    label_dir = Path(args.label_path)
    target_dir = Path(args.target_path)
    
    # 创建标签文件夹
    target_dir.mkdir(exist_ok=True)
    
    threads = []  # 保存线程的list
    pbar = tqdm(total_images)  # for every image file
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

    # 输出统计信息
    ptab = PrettyTable(['Item', 'Number'])
    ptab.add_row(['Total images', COUNTER["images"]])
    ptab.add_row(['Converted', COUNTER["found"]])
    ptab.add_row(['Missing label', COUNTER["missing"]])
    ptab.add_row(['Corrupt', COUNTER["corrupt"]])
    ptab.add_row(['Incomplete points', COUNTER["incomplete_pts"]])
    ptab.add_row(['Out of boundary', COUNTER["out_of_boundary"]])
    ptab.add_row(['Background', COUNTER["background"]])
    ptab.add_row(['Skip existed target file', COUNTER["skip"]])
    ptab.add_row(['Processed', COUNTER["touch"]])
    
    LOGGER.info(ptab)
    
    if COUNTER["found"] + COUNTER["background"] + COUNTER['skip'] + COUNTER['missing'] == COUNTER["images"]:
        LOGGER.info(colorstr('green', 'bold', '✅ All conversion has done correctly!'))
        if COUNTER['missing'] != 0:
            LOGGER.warning(colorstr('yellow', 'bold', f"⚠️ There are {COUNTER['missing']} images without label, and they have be regarded as negative samples!"))
    else:
        LOGGER.warning(colorstr('red', 'bold', "⚠️ Some question have occurred, please check dataset!"))

    if COUNTER['skip'] == COUNTER['images']:
        LOGGER.warning(f"⚠️ All target file have been skipped, please check dataset!")

    LOGGER.info(f"⏳ The cost time of {str(FILE.name)} is {colorstr(calc_cost_time(t1, time.time()))}")
    LOGGER.info(f"👀 The detail information has saved at {colorstr(LOGGER.handlers[0].baseFilename)}")
