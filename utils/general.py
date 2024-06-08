import contextlib
import logging
import datetime
from pathlib import Path
from typing import Union
from PIL import ExifTags, Image, ImageOps
from lxml import etree
from datetime import timedelta
from prettytable import PrettyTable
from xml.dom.minidom import Document


IMAGE_TYPE = ['.png', '.jpg', '.jpeg', '.bmp', 'webp']

RECORDER = {
    'corrupt': 0,  # 图片破损的数量
    'incomplete_pts': 0,  # 标签点的数量不为4
    'out_of_boundary': 0,  # 坐标点越界
    'reversed': 0,  # 坐标点反了
    'skip': 0,  # 目标文件存在，跳过的数量
    'touch': 0,  # 触摸过的图片数量
    'found': 0,  # 完成转换的标签
    'missing': 0,  # 缺少标签的图片数量
    'background': 0,  # 负样本图片的数量
    'objects': 0,  # 对象总数
}

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        # 💡 注意：此时orientation变量也会被保留
        #        （在Python的for循环中，那个变量不是临时变量，是可以跳出作用域被使用的）
        break  


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


def get_logger(file) -> logging.Logger:
    # 定义日志保存路径
    current_time = datetime.datetime.now()  
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")  # e.g. '20240606111504'
    script_path = Path(file)
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


def listdir(path: Union[Path, str], extension: Union[str, list, tuple]=IMAGE_TYPE) -> list:
    """遍历一下目录下的指定文件

    Args:
        path (Union[Path, str]): 文件夹路径
        extension (Union[str, list, tuple], optional): 需要的文件后缀. Defaults to 'IMAGE_TYPE -> ['.png', '.jpg', '.jpeg', '.bmp', 'webp']'.

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


def second_confirm(script: Path = None, LOGGER: logging.Logger = None):
    script_name = str(script.name) if script else 'script'
    user_name = str(script.owner()) if script else Path.cwd().owner()

    msg = colorstr("Please enter 'yes' (y) to continue, or enter anything else to stop the program: ")
    LOGGER.info(msg) if LOGGER else print(msg)

    user_input = input(colorstr('bold', '>>>> '))
    if user_input.lower() not in ('yes', 'y', 'continue'):
        msg = colorstr(
            'red', 'bold',
            f"The input of User({user_name}) is: {user_input}\n"
            f"The {script_name} is cancelled!"
        )
        LOGGER.critical(msg) if LOGGER else print(msg)
        exit(1)
    else:
        msg = colorstr(
            'green', 'bold',
            f"The input of User({user_name}) is: {user_input}\n"
            f"The {script_name} will execute."
        )
        LOGGER.info(msg) if LOGGER else print(msg)

        
def verify_image(image: Path, LOGGER: logging.Logger = None) -> bool:
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
        msg = f"⚠️  The size of {str(image.name)} ({shape[0]}×{shape[1]}) is less than 10×10!"
        LOGGER.warning(msg) if LOGGER else print(msg)
    
    # 如果图片的格式是JPEG
    if im.format.lower() in ("jpg", "jpeg"):
        with open(image, "rb") as f:  # 使用read-binary的方式打开JEPG图片
            f.seek(-2, 2)  # 将文件指针从文件末尾向后移动 2 个字节
        
            # 判断这张JPEG图片是否是破损的
            if f.read() != b"\xff\xd9":  # corrupt JPEG
                msg = f"⚠️  {image} is a corrupt image!"
                LOGGER.warning(msg) if LOGGER else print(msg)
                return False
    return True


def exif_size(img):
    """返回经过EXIF校正的PIL尺寸。

    Args:
        img (PIL.Image): PIL图像对象。

    Returns:
        tuple: 包含宽度（width）和高度（height）的元组。
    """
    # 获取原始尺寸
    s = img.size  # (width, height)
    
    # 尝试获取EXIF信息（如果EXIF信息不可用或不存在，则不进行旋转）
    with contextlib.suppress(Exception):
        # 提取旋转信息
        rotation = dict(img._getexif().items())[orientation]
        
        # 检查是否需要旋转
        if rotation in [6, 8]:  # 旋转270度或90度
            s = (s[1], s[0])  # 交换宽度和高度
            
    # 返回校正后的尺寸
    return s


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


def read_txt(txt: Path) -> list:
    if isinstance(txt, str):
        txt = Path(txt)

    with txt.open('r') as f:
        lines = f.readlines()
    
    return [line.strip() for line in lines]

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


def xyxy2xywh(x1, y1, x2, y2) -> tuple:
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return x, y, w, h


def xywh2xyxy(x, y, w, h) -> tuple:
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return x1, y1, x2, y2


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


def check_dataset(num_images, num_labels, LOGGER: logging.Logger=None) -> None:
    assert num_images > 0, colorstr('red', 'bold', "❌ The number of images is 0, it's illegal!")
    assert num_labels > 0, colorstr('red', 'bold', "❌ The number of labels is 0, it's illegal!")

    if num_images > num_labels:
        msg = f"⚠️ The number of image ({num_images}) > labels ({num_labels}), \
            the image without label file will be regarded as {colorstr('red', 'bold', 'negative!')}"
        LOGGER.warning(msg) if LOGGER else print(msg)
    if num_images < num_labels:
        msg = f"⚠️ The number of image ({num_images}) < labels ({num_labels}), \
            there are {num_labels - num_images} redundant label file."
        LOGGER.warning(msg) if LOGGER else print(msg)


def dict2table(d: dict, field_names=['Key', 'Value'], align='r', replace_keys: dict = {}, ommit_empty: bool = False) -> PrettyTable:
    """将一个字典转换为PrettyTable对象

    Args:
        d (dict): 传入的字典
        field_names (list): PrettyTable的列名. Defaults to ['Key', 'Value'].
        align (str, optional): PrettyTable的对齐方式（'l', 'c', 'r'）. Defaults to 'r'.
        replace_keys (dict): 需要替换显示的键. Defaults to {}.
        ommit_empty (bool): 如果字典中的value为int且为0，则不显示该键值对. Defaults to False.

    Returns:
        PrettyTable: 返回一个PrettyTable对象
    """
    assert isinstance(d, dict), f"❌ 这个函数需要传入一个dict而非{type(d)}!"

    # 替换显示的键
    d2 = {} if replace_keys else d
    for k, v in d.items():
        new_key = replace_keys.get(k, k)  # 即便对应的键不存在也不会丢失之前的键值对
        d2[new_key] = v
                
    ptab = PrettyTable(field_names)
    ptab.align = align

    # 处理 ommit_empty
    d3 = {k: v for k, v in d2.items() if not (ommit_empty and isinstance(v, int) and v == 0)}

    for k, v in d3.items():
        # 处理 'classes_dict'
        if isinstance(k, str) and k.lower() == 'classes_dict':
            # 防呆设计：classes_dict是一个list，则转换为dict
            # classes_dict形式：{0: 'cat', 1: 'dog'}
            v = {idx: name for idx, name in enumerate(v)} if isinstance(v, list) else v
            for idx, class_name in v.items():
                ptab.add_row([f"  class-{idx}", class_name])  # print class_dict, e.g.   class-0: 'cat'
        else:
            ptab.add_row([k, v])
    
    return ptab


def reverse_dict(d: dict):
    return {v: k for k, v in d.items()}


def statistics(recorder: dict) -> dict:
    objects = recorder.get('objects', 0)  # 对象总数
    samples = recorder.get('images', 0)  # 总共的图片数量
    positives = recorder.get('found', 0)  # 正样本数量
    negatives = recorder.get('missing', 0) + recorder.get('background', 0) # 负样本数量

    recorder['average objects for all'] = (objects // samples) if samples > 0 else 'N/A'
    recorder['average objects for positives'] = (objects // positives) if positives > 0 else 'N/A'
    recorder['positive ratio'] = (f"{round(positives / samples * 100, 2)}%") if samples > 0 else 'N/A'
    recorder['negative ratio'] = (f"{round(negatives / samples * 100, 2)}%") if samples > 0 else 'N/A'

    return recorder
    

class XMLRecorder():
    def __init__(self, image: Path, img_w: int, img_h: int, img_c: Union[int, str]):
        if not isinstance(image, Path):
            self.image = Path(image)

        self.xmlBuilder = Document()  # 创建一个 XML 文档构建器
        self.annotation = self.xmlBuilder.createElement("annotation")  # 创建annotation标签
        self.xmlBuilder.appendChild(self.annotation)

        # folder标签
        folder = self.xmlBuilder.createElement("folder")  
        foldercontent = self.xmlBuilder.createTextNode('images')
        folder.appendChild(foldercontent)
        self.annotation.appendChild(folder)  # folder标签结束

        # filename标签
        filename = self.xmlBuilder.createElement("filename")  
        filenamecontent = self.xmlBuilder.createTextNode(str(image.name))
        filename.appendChild(filenamecontent)
        self.annotation.appendChild(filename)  # filename标签结束

        # size标签
        size = self.xmlBuilder.createElement("size")  
        width = self.xmlBuilder.createElement("width")  # size子标签width
        widthcontent = self.xmlBuilder.createTextNode(str(img_w))
        width.appendChild(widthcontent)
        size.appendChild(width)  # size子标签width结束

        height = self.xmlBuilder.createElement("height")  # size子标签height
        heightcontent = self.xmlBuilder.createTextNode(str(img_h))
        height.appendChild(heightcontent)
        size.appendChild(height)  # size子标签height结束

        depth = self.xmlBuilder.createElement("depth")  # size子标签depth
        depthcontent = self.xmlBuilder.createTextNode(str(img_c))
        depth.appendChild(depthcontent)
        size.appendChild(depth)  # size子标签depth结束
        self.annotation.appendChild(size)  # size标签结束


    def add_object(self, class_name, x1, y1, x2, y2):
        # 创建<object>标签
        object = self.xmlBuilder.createElement("object") 

        # 1. name标签
        picname = self.xmlBuilder.createElement("name")  
        namecontent = self.xmlBuilder.createTextNode(class_name)  # 确定是哪个类别
        picname.appendChild(namecontent)
        object.appendChild(picname)  # name标签结束

        # 2. pose标签
        pose = self.xmlBuilder.createElement("pose")  
        posecontent = self.xmlBuilder.createTextNode("Unspecified")
        pose.appendChild(posecontent)
        object.appendChild(pose)  # pose标签结束

        # 3. truncated标签
        truncated = self.xmlBuilder.createElement("truncated")  
        truncatedContent = self.xmlBuilder.createTextNode("0")
        truncated.appendChild(truncatedContent)
        object.appendChild(truncated)  # truncated标签结束
        
        # 4. difficult标签
        difficult = self.xmlBuilder.createElement("difficult")  
        difficultcontent = self.xmlBuilder.createTextNode("0")
        difficult.appendChild(difficultcontent)
        object.appendChild(difficult)  # difficult标签结束

        # 5. bndbox标签
        bndbox = self.xmlBuilder.createElement("bndbox")  
        # 5.1 xmin标签
        xmin = self.xmlBuilder.createElement("xmin")  
        mathData = x1
        xminContent = self.xmlBuilder.createTextNode(str(mathData))
        xmin.appendChild(xminContent)
        bndbox.appendChild(xmin)  # xmin标签结束

        # 5.2 ymin标签
        ymin = self.xmlBuilder.createElement("ymin")  # ymin标签
        mathData = y1
        yminContent = self.xmlBuilder.createTextNode(str(mathData))
        ymin.appendChild(yminContent)
        bndbox.appendChild(ymin)  # ymin标签结束
        
        # 5.3 xmax标签
        xmax = self.xmlBuilder.createElement("xmax")  # xmax标签
        mathData = x2
        xmaxContent = self.xmlBuilder.createTextNode(str(mathData))
        xmax.appendChild(xmaxContent)
        bndbox.appendChild(xmax)  # xmax标签结束

        # 5.4 ymax标签
        ymax = self.xmlBuilder.createElement("ymax")  # ymax标签
        mathData = y2
        ymaxContent = self.xmlBuilder.createTextNode(str(mathData))
        ymax.appendChild(ymaxContent)
        bndbox.appendChild(ymax)  # ymax标签结束

        object.appendChild(bndbox)  # bndbox标签结束
        self.annotation.appendChild(object)  # object标签结束

    def save(self, target_path: Path):
        with target_path.open('w') as f:
            self.xmlBuilder.writexml(
                f, 
                indent='\t', 
                newl='\n',
                addindent='\t', 
                encoding='utf-8'
            )
