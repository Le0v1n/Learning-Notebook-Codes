import logging
import datetime
from pathlib import Path
from typing import Union
from PIL import Image
from lxml import etree
from datetime import timedelta


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


def get_logger() -> logging.Logger:
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


# 创建日志 -> global
LOGGER = get_logger()


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
    LOGGER.info(colorstr("Please enter 'yes' (y) to continue, or enter anything else to stop the program: "))
    
    user_input = input(colorstr('bold', '>>>> '))
    if user_input.lower() not in ('yes', 'y', 'continue'):
        LOGGER.info(colorstr('red', 'bold', f"The script is cancelled by {Path.cwd().owner()}!"))
        exit()
        
        
def verify_image(image: Path) -> bool:
    if isinstance(image, str):
        image = Path(image)
        
    im = Image.open(image)
    try:
        im.verify()  # PIL verify --> 验证图像文件的完整性。如果有问题则报错，会被except捕获
    except:
        return False
    
    # 检查图片尺寸（高度和宽度最小为10）
    shape = im.size
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


def split_list_equally(lst, n):
    size = len(lst) // n  # 计算每份的大小
    remainder = len(lst) % n  # 计算剩余的元素数量
    
    # 使用列表切片来分割列表
    result = [lst[i*size:(i+1)*size] for i in range(n)]
    
    # 将剩余的元素分配到最后一份列表中
    if remainder > 0:
        result[-1].extend(lst[-remainder:])
    
    return result


def calc_cost_time(t1: float, t2: float) -> str:
    # 计算时间差
    t = t2 - t1
    # 确保时间差是正数
    assert t >= 0, f"❌  There occur an error about time(cost time({t}) < 0), the start time is: {t1}, and the end time is: {t2}."
    
    # 使用 timedelta 将时间差转换为时分秒
    td = timedelta(seconds=t)
    
    # 提取小时、分钟和秒
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # 格式化输出
    return f"{hours}h {minutes}m {seconds}s"


def dataset_number_comparison(num_images, num_labels) -> None:
    if num_images > num_labels:
        LOGGER.warning(
            f"⚠️ The number of image ({num_images}) > labels ({num_labels}), "
            f"the image without label file will be regarded as {colorstr('red', 'bold', 'negative!')}"
        )
    if num_images < num_labels:
        LOGGER.warning(
            f"⚠️ The number of image ({num_images}) < labels ({num_labels}), "
            f"there are {num_labels - num_images} redundant label file."
        )