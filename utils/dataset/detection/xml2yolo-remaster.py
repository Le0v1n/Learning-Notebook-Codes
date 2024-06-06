import sys
import math
import datetime
import argparse
import logging
import threading
from pathlib import Path
from typing import Union
from tqdm import tqdm
from lxml import etree
from PIL import Image
from prettytable import PrettyTable


ROOT = Path.cwd().resolve()
FILE = Path(__file__).resolve()  # 当前脚本的绝对路径
if str(ROOT) not in sys.path:  # 解决VSCode没有ROOT的问题
    sys.path.append(str(ROOT))
ROOT = ROOT.relative_to(Path.cwd())

# 创建一个计数器字典，global
COUNTER = {
    'missing': 0,  # 缺少标签的图片数量
    'corrupt': 0,  # 图片破损的数量
    'background': 0,  # 负样本图片的数量
    'incomplete_pts': 0,  # 标签点的数量不为4
    'pass': 0,  # 完成的数量
}


def get_logger() -> logging:
    # 定义日志保存路径
    current_time = datetime.datetime.now()  
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")  # e.g. '20240606111504'
    script_path = Path(__file__)
    log_dir = script_path.parent.joinpath('logs')
    log_filepath = log_dir.joinpath(script_path.stem + '-' + formatted_time + '.log')  # e.g. 'utils/dataset/detection/logs/xml2yolo-remaster-20240606112020.log'

    # 创建日志的父级文件夹
    log_dir.mkdir(exist_ok=True)
    
    # 配置日志输出的格式和级别
    logging.basicConfig(
        filename=log_filepath, 
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 创建日志记录器
    logger = logging.getLogger()

    # 创建控制台处理器并添加到日志记录器
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    
    # 声明log的位置
    logger.info(f"The log file has create at {str(log_filepath)}")
    
    return logger


# 创建日志
LOGGER = get_logger()


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]

    
def listdir(path: Union[Path, str], extension: Union[str, list, tuple]='.png') -> list:
    """遍历一下目录下的指定文件

    Args:
        path (Union[Path, str]): 文件夹路径
        extension (Union[str, list, tuple], optional): 需要的文件后缀. Defaults to '.png'.

    Returns:
        list: 返回一个list，里面是所有符合要求的文件路径
    """
    if isinstance(path, str):
        path = Path(path)
    if isinstance(extension, (tuple, list)):
        files = []
        for ext in extension:
            for file in path.glob(f"*{ext}"):
                files.append(file)
    else:
        files = [file for file in path.glob(f"*{extension}")]  
    
    return files


def second_confirm():
    LOGGER.info(colorstr("⌨️  Please enter 'yes' (y) to continue, or enter anything else to stop the program: "))
    user_input = input(colorstr('bold', '>>>> '))
    if user_input.lower() not in ('yes', 'y', 'continue'):
        LOGGER.info(colorstr('red', f'Cancelled by {FILE.owner()}!'))
        exit()
        
        
def verify_image(image: Path) -> bool:
    if isinstance(image, str):
        image = Path(image)
        
    im = Image.open(image)
    im.verify()  # PIL verify --> 验证图像文件的完整性。如果有问题则报错，会被except捕获
    shape = im.size
    
    # 检查图片尺寸（高度和宽度最小为10）
    if shape[0] < 10 or shape[1] < 10:
        LOGGER.warning(f"⚠️  The size of {str(image.name)} ({shape[0]}×{shape[1]}) is less than 10×10!")
    
    # 如果图片的格式是JPEG
    if im.format.lower() in ("jpg", "jpeg"):
        with open(image, "rb") as f:  # 使用read-binary的方式打开JEPG图片
            f.seek(-2, 2)  # 将文件指针从文件末尾向后移动 2 个字节
        
            # 判断这张JPEG图片是否是破损的
            if f.read() != b"\xff\xd9":  # corrupt JPEG
                LOGGER.warning(f"⚠️  {image} is a corrupt image!")
                return False
    return True


def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args：
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def read_xml(xml: Path) -> dict:
    if isinstance(xml, str):
        xml = Path(xml)
        
    with xml.open('r') as f:
        xml_str = f.read()
        
    # 将XML字符串编码为字节序列
    xml_bytes = xml_str.encode('utf-8')

    # 使用lxml解析字节序列的XML数据
    xml = etree.fromstring(xml_bytes)
    
    return parse_xml_to_dict(xml)["annotation"]


def fix_illegal_coordinates(xmin, ymin, xmax, ymax, img_width, img_height) -> tuple:
    """
    修复不合法的坐标（非负且xmin <= xmax，ymin <= ymax）。

    Parameters:
        xmin (float): 左上角 x 坐标
        ymin (float): 左上角 y 坐标
        xmax (float): 右下角 x 坐标
        ymax (float): 右下角 y 坐标
        
    Returns:
        xmin, ymin, xmax, ymax, msg
    """
    msg = []
    if xmin < 0: 
        msg.append(f'xmin({xmin:.4f}) < 0')
        xmin = 0.0
    if ymin < 0: 
        msg.append(f'ymin({ymin:.4f}) < 0')
        ymin = 0.0
    if xmax < 0: 
        msg.append(f'xmax({xmax:.4f}) < 0')
        xmax = 0.0
    if ymax < 0: 
        msg.append(f'ymax({ymax:.4f}) < 0')
        ymax = 0.0

    # 如果出现越界
    if xmin > img_width: 
        msg.append(f"xmin({xmin:.4f}) > width({img_width})")
        xmin = float(img_width)
    if ymin > img_height: 
        msg.append(f"ymin({ymin:.4f}) > height({img_height})")
        ymin = float(img_height)
    if xmax > img_width: 
        msg.append(f"xmax({xmax:.4f}) > width({img_width})")
        xmax = float(img_width)
    if ymax > img_height: 
        msg.append(f"ymax({ymax:.4f}) > height({img_height})")
        ymax = float(img_height)
    
    return xmin, ymin, xmax, ymax, msg


def fix_reverse_coordinates(xmin, ymin, xmax, ymax) -> tuple:
    msg = []
    if xmin > xmax or ymin > ymax:  # 出现错误
        if xmin > xmax:
            msg.append(f"xmin({xmin:.4f}) > xmax({xmax:.4f})")
        if ymin > ymax:
            msg.append(f"ymin({ymin:.4f}) > ymax({ymax:.4f})")
            
        # 重新计算中心点坐标
        xcenter, ycenter = (xmin + xmax) / 2, (ymin + ymax) / 2
        
        # 根据中心点坐标(xcenter, ycenter)重新计算左上角坐标(xmin, ymin)和右上角坐标(xmax, ymax)
        width, height = abs(xmax - xmin), abs(ymax - ymin)
        
        # 计算和更新坐标
        xmin, ymin = xcenter - width / 2, ycenter - height / 2
        xmax, ymax = xcenter + width / 2, ycenter + height / 2

    return xmin, ymin, xmax, ymax, msg


def xyxy2xywh(x1, y1, x2, y2):
    x = x1 + (x2 - x1) / 2
    y = y1 + (y2 - y1) / 2
    w = x2 - x1
    h = y2 - y1
    return x, y, w, h


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, default="Datasets/coco128/train/images", help="图片路径")
    parser.add_argument("--xml-path", type=str, default="Datasets/coco128/train/annotations-xml", help="xml标签路径")
    parser.add_argument("--yolo-path", type=str, default="Datasets/coco128/train/labels", help="yolo标签路径保存路径")
    parser.add_argument("--classes", type=str, nargs='+', default=['cat', 'dog'], help="数据集标签")
    parser.add_argument("--image-format", type=str, nargs='+', default=['.png', '.jpg', '.jpeg', '.bmp', 'webp'], help="允许的图片格式")
    parser.add_argument("--override", action='store_true', default=False, help="如果对应的.txt文件存在，是否覆盖它")
    parser.add_argument("--num-threading", type=int, default=4, help="使用的线程数，不使用多线程则设置为1")
    
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(args: argparse, images: list, pbar: tqdm):
    # found, missing, corrupt, incomplete_pts, background
    nf, nm, nc, ni, nb = 0, 0, 0, 0, 0
    for image in images:  # image: PosixPath
        # 更新进度条信息
        pbar.set_description(f"Processing {colorstr(image.name):<30s}")
        
        # 获取图片尺寸
        img_width, img_height = Image.open(image).size
        if not verify_image(image):  # 验证图片是否破损
            COUNTER["corrupt"] += 1
            pbar.update()
            continue
        
        # 打开应对的xml文件，得到一个字典
        xml = xml_dir.joinpath(image.stem + '.xml')
        if not xml.exists():
            LOGGER.error(f"❌ {str(image)} don't find the corresponding xml file!")
            COUNTER["missing"] += 1
            pbar.update()
            continue      
        xml_data = read_xml(xml)
        
        # 处理.txt
        yolo = yolo_dir.joinpath(image.stem + '.txt')
        if yolo.exists() and not yolo.read_text():  # 文件存在且不为空
            if args.override:  # 覆盖掉之前的txt内容
                LOGGER.warning(f"⚠️  {args.override = }, the content of {str(yolo)} will be overrode!")
            else:  # 不覆盖
                LOGGER.info(f"The {str(yolo)} file will not be overrode because {args.override = }.")
                pbar.update()
                continue
            
        with yolo.open('w') as f:
            objects = xml_data.get("object", None)
            if not objects:  # Negative samples
                COUNTER['background'] += 1
                pbar.update()
                continue
            
            # Positive samples
            for index, obj in enumerate(xml_data["object"]):
                # 检查是否有缺点的情况（坐标点的数量不为4）
                num_pts = len(obj["bndbox"])
                if num_pts != 4:
                    LOGGER.error(f"❌ The {index} object of {str(xml)} has incomplete points({num_pts} < 4)")
                    COUNTER['incomplete_pts'] += 1
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
                    LOGGER.warning(f"⚠️ The {index} object of {str(xml)} has illegal coordinates: {msg}")
                
                # 修复相反的坐标：x2y2x1y1 -> x1y1x2y2
                x1, y1, x2, y2, msg = fix_reverse_coordinates(x1, y1, x2, y2)
                if msg:
                    msg = [f"[{i}] {content}" for i, content in enumerate(msg)]
                    msg = ", ".join(msg)
                    LOGGER.warning(f"⚠️ The {index} object of {str(xml)} has illegal coordinates: {msg}")
                
                # 获取对应的类别并转换为索引
                class_name = obj["name"]
                try:
                    class_index = classes_dict[class_name]
                except:
                    LOGGER.error(f"❌ {class_name} of {str(xml)} don't exist in {classes_dict}!")
                    exit()

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
        COUNTER["pass"] += 1
        pbar.update()
    pbar.close()
    
    # 输出统计信息
    ptab = PrettyTable(['Item', 'Description'])
    ptab.add_row(['Images', COUNTER["images"]])
    ptab.add_row(['Converted', COUNTER["pass"]])
    ptab.add_row(['Missing xml', COUNTER["missing"]])
    ptab.add_row(['Corrupt image', COUNTER["corrupt"]])
    ptab.add_row(['Incomplete points', COUNTER["incomplete_pts"]])
    ptab.add_row(['Background images', COUNTER["background"]])
    
    LOGGER.info(ptab)
    
    if COUNTER["pass"] + COUNTER["background"] == COUNTER["images"]:
        LOGGER.info(f"✅ Conversion has done!")
    else:
        LOGGER.warning(f"⚠️ Some question have occurred, check dataset please!")
        
        
def split_list_equally(lst, n):
    size = len(lst) // n  # 计算每份的大小
    remainder = len(lst) % n  # 计算剩余的元素数量
    
    # 使用列表切片来分割列表
    result = [lst[i*size:(i+1)*size] for i in range(n)]
    
    # 将剩余的元素分配到最后一份列表中
    result[-1].extend(lst[-remainder:])
    
    return result
        

if __name__ == "__main__":
    args = parse_opt()
    
    # 读取所有的图片和标签
    images = listdir(args.image_path, extension=args.image_format)
    labels = listdir(args.xml_path, extension='.xml')
    COUNTER['images'] = len(images)
    COUNTER['labels'] = len(labels)
    
    # 创建类别字典
    classes_dict = {cla: i for i, cla in enumerate(args.classes)}  # str: int, e.g. {'cat': 0, 'dog': 1}
    COUNTER['nc'] = len(args.classes)
    if COUNTER['images'] > COUNTER['xmls']:
        LOGGER.warning(
            f"❌ Image number ({COUNTER['images']}) > XML number({COUNTER['xmls']}),"
            f"the image without xml file will be regarded as {colorstr('red', 'bold', 'negative!')}"
        )
    
    ptab = PrettyTable(['参数', '详情'])
    ptab.align = 'l'
    ptab.add_row(['图片路径', args.image_path])
    ptab.add_row(['图片数量', COUNTER['images']])
    ptab.add_row(['XML路径', args.xml_path])
    ptab.add_row(['XML数量', COUNTER['xmls']])
    ptab.add_row(['类别数', COUNTER['nc']])
    ptab.add_row(['类别', ''])
    for i, cla in classes_dict.items():
        ptab.add_row([f"    class-{i}", cla])
    LOGGER.info(ptab)
    
    # 2FA
    second_confirm()
    
    # 创建Path对象
    xml_dir = Path(args.xml_path)
    yolo_dir = Path(args.yolo_path)
    
    # 创建标签文件夹
    yolo_dir.mkdir(exist_ok=True)
    
    # 根据线程数，得到每个线程需要处理的图片list
    total_image_lists = split_list_equally(images, args.num_threading)
    
    threads = []  # 保存线程的list
    pbar = tqdm(images)  # for every image file
    for images in total_image_lists:
        t = threading.Thread(
            target=main, 
            args=(
                args, 
                images,
                pbar,
            )
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
