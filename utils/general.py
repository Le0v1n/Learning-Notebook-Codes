import argparse
import contextlib
import logging
import datetime
import json
from pathlib import Path
from typing import Union
from PIL import ExifTags, Image
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
    'objects': 0,  # 对象总数,
    'gray': 0,  # 灰度图数量
    'RGBA': 0,  # RGBA图片的数量
}

TranslationDict = {
    'found': '正样本(有标签)数量',
    'missing': '负样本(没有标签)数量',
    '❌ corrupt': '破损的图片数量',
    '❌ illegal_pts': '坐标点个数≠4的数量',
    '❌ out_of_boundary': '坐标越界的数量',
    '❌ reversed': '坐标点反了的数量',
    'background': '负样本(有空标签)的数量',
    'touch': '程序touch过的数量',
    'skip': '跳过(目标标签存在)的数量',
    'image path': '图片路径',
    'label path': '标签路径',
    'target path': '保存路径',
    'images': '图片的数量',
    'labels': '标签的数量',
    'ndigit': '坐标保存小数点位数',
    'nc': '类别数',
    'num classes': '类别数',
    'threadings': '使用的线程数',
    'data num of every threading': '单个线程的并发量',
    'script': '脚本名称',
    'objects': '对象总数',
    'average objects for all': '平均每张图片的对象数',
    'average objects for positives': '平均每张正样本图片的对象数',
    'positive ratio': '正样本比例',
    'negative ratio': '负样本比例',
    'label not exist': '图片对应标签不存在的数量',
    '❌ label issue': '标签有问题的数量',
    'label format': '标签格式',
    'redundant': '冗余标签（没有对应图片）的数量',
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


def second_confirm(msg: str = None, script: Path = None, LOGGER: logging.Logger = None):
    script_name = str(script.name) if script else 'script'
    user_name = str(script.owner()) if script else Path.cwd().owner()

    msg = colorstr("Please enter 'yes' (y) to continue, or enter anything else to stop the program: ") if not msg else msg
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
    
    # 检查图片通道数
    img_channel = 3 if im.mode == "RGB" else 1 if im.mode == "L" else 4 if im.mode == "RGBA" else "Unknown"
    if img_channel == 1:
        msg = f"⚠️ [Gray image] Found gray image! -> {str(image)}"
        LOGGER.warning(msg) if LOGGER else print(msg)
        RECORDER["gray"] += 1
    elif img_channel == 4:
        msg = f"⚠️ [RGBA image] Found RGBA image! -> {str(image)}"
        LOGGER.warning(msg) if LOGGER else print(msg)
        RECORDER["RGBA"] += 1
    elif img_channel == "Unknown":
        msg = f"❌ [Unknown channel image] Found unknown channel image! -> {str(image)}"
        LOGGER.warning(msg) if LOGGER else print(msg)
        RECORDER["RGBA"] += 1
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
    xml = Path(xml)
        
    with xml.open('r') as f:
        xml_str = f.read()
        
    # 将XML字符串编码为字节序列
    xml_bytes = xml_str.encode('utf-8')

    # 使用lxml解析字节序列的XML数据
    xml = etree.fromstring(xml_bytes)
    
    return parse_xml_to_dict(xml)["annotation"]


def read_txt(txt: Path) -> list:
    txt = Path(txt)
        
    with txt.open('r') as f:
        lines = f.readlines()
    
    return [line.strip() for line in lines]


def read_json(jsonfile: Path) -> dict:
    with jsonfile.open('r') as f:
        return json.load(f)


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
        msg = f"⚠️ The number of image ({num_images}) > labels ({num_labels}), " \
              f"the image without label file will be regarded as {colorstr('red', 'bold', 'negative!')}"
        LOGGER.warning(msg) if LOGGER else print(msg)
    if num_images < num_labels:
        msg = f"⚠️ The number of image ({num_images}) < labels ({num_labels}), " \
              f"there are {num_labels - num_images} redundant label file."
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


class TXTWriter():
    def __init__(self):
        self.objects = []
    

    def add_object(self, class_id, x, y, w, h):
        # 💡 强制保留6位小数
        self.objects.append([str(f"{int(class_id)}"), f"{x:.6f}", f"{y:.6f}", f"{w:.6f}", f"{h:.6f}"])

    
    def save(self, target_path: Path):
        with target_path.open('w') as f:  # 一次性将所有的objects都写入txt
            for i, obj_data in enumerate(self.objects):
                # 使用join将列表转换为字符串，并用空格分隔
                line = " ".join(obj_data)
                # 写入文件，如果i不是0，则在前面添加一个换行符
                f.write(line if i == 0 else '\n' + line)
    

class XMLWriter():
    def __init__(self, image: Path, img_w: int, img_h: int, img_c: Union[int, str]):
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
        xminContent = self.xmlBuilder.createTextNode(str(x1))
        xmin.appendChild(xminContent)
        bndbox.appendChild(xmin)  # xmin标签结束

        # 5.2 ymin标签
        ymin = self.xmlBuilder.createElement("ymin")  # ymin标签
        yminContent = self.xmlBuilder.createTextNode(str(y1))
        ymin.appendChild(yminContent)
        bndbox.appendChild(ymin)  # ymin标签结束
        
        # 5.3 xmax标签
        xmax = self.xmlBuilder.createElement("xmax")  # xmax标签
        xmaxContent = self.xmlBuilder.createTextNode(str(x2))
        xmax.appendChild(xmaxContent)
        bndbox.appendChild(xmax)  # xmax标签结束

        # 5.4 ymax标签
        ymax = self.xmlBuilder.createElement("ymax")  # ymax标签
        ymaxContent = self.xmlBuilder.createTextNode(str(y2))
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


class JsonWriter():
    def __init__(self, image: Path, img_w: int, img_h: int, version="5.4.1"):
        self.image = Path(image)

        # 创建 Json 文件的内容
        self.json_dict = {
            "version": version,
            "flags": {},
            "shapes": [],
            "imagePath": str(image.name),
            "imageData": None,
            "imageHeight": img_h,
            "imageWidth": img_w
        }

    
    def add_object(self, class_name, x1, y1, x2, y2):
        self.json_dict["shapes"].append(
            {
                "label": class_name,
                "points": [
                    [x1, y1],
                    [x2, y2]
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None
            }
        )

    
    def save(self, target_path: Path):
        with target_path.open('w', encoding='utf-8') as f:
            json.dump(
                self.json_dict, 
                f,
                ensure_ascii=False,
                indent=2
            ) 


def fix_label_format(label_format: str) -> str:
    if isinstance(label_format, str):
        if '.' not in label_format:  # 如果没有.则添加
            label_format = '.' + label_format

        if label_format.lower() in ('.txt', '.yolo'):
            return '.txt'
        elif label_format.lower() in ('.json', ):
            return '.json'
        elif label_format.lower() in ('.xml', ):
            return '.xml'
        else:
            raise NotImplementedError(
                f"❌ The current script only supports label with {colorstr('.txt, .json, .xml')}, "
                f"and does not support {colorstr(label_format)}!"
            )
    else:
        raise TypeError(f"❌ The type of label_format should be {colorstr('str')} instead of {colorstr(type(label_format))}!")
    

class LabelVerifier():
    def __init__(self, image: Path, label: Path, classes_dict: int, img_width: int, img_height: int, img_channel: int):
        self.image = Path(image)
        self.label = Path(label)
        self.classes_dict = classes_dict
        self.img_width = img_width
        self.img_height = img_height
        self.img_channel = img_channel
        self.msgs = []  # 存放报错信息用的
        self.count = 1

        # 获取标签的后缀
        self.suffix = self.label.suffix


    def log(self, msg: str):
        self.count += 1
        self.msgs.append(msg)        


    def objects_exist(self) -> bool:
        # 如果没有object -> 定义为负样本
        if not self.objects:
            self.log(f"[{self.count}] There is no any objects.")
            return False
        return True

    
    def verify_coordinates_for_object(self, x1, y1, x2, y2) -> None:
        # 检查不合规的坐标：负数和越界
        msg = fix_illegal_coordinates(x1, y1, x2, y2, self.img_width, self.img_height)[-1]
        if msg:
            self.log(f"[{self.count}] The object has negative number or out of boundary {msg}.")

        # 检查相反的坐标：x2y2x1y1 -> x1y1x2y2
        msg = fix_reverse_coordinates(x1, y1, x2, y2)[-1]
        if msg:
            self.log(f"[{self.count}] The object of has reversed coordinates {msg}.")


    def verify_num_coordinates_for_object(self, object_info) -> bool:
        if self.suffix == '.txt':
            object_info = object_info.split(' ')
            # 将坐标点转换为一个list
            num_pts = len(object_info[1:])
        elif self.suffix == '.xml':
            # 先判断有没有<bndbox>
            if not object_info.get('bndbox', None):
                self.log(f"[{self.count}] The object don't have 'bndbox').")
                return False
            
            # 将坐标点转换为一个list
            num_pts = len(object_info["bndbox"])

        elif self.suffix == '.json':
            # # 将坐标点转换为一个list
            object_info['points'] = [coordinate for pair in object_info['points'] for coordinate in pair]
            num_pts = len(object_info["points"])

        # 检查：坐标点的个数是否为4
        if num_pts != 4:
            self.log(f"[{self.count}] The object has illegal points({num_pts} != 4).")
            return False
        return True


    def normalize_coordinates_for_object(self, object_info) -> tuple:
        if self.suffix == '.txt':
            object_info = object_info.split(' ')

            # 获取每个object的box信息
            x = float(object_info[1])
            y = float(object_info[2])
            w = float(object_info[3])
            h = float(object_info[4])

            # xywh -> xyxy
            x1, y1, x2, y2 = xywh2xyxy(x, y, w, h)

            # 坐标映射回原图大小
            x1 = round(x1 * self.img_width)
            y1 = round(y1 * self.img_height)
            x2 = round(x2 * self.img_width)
            y2 = round(y2 * self.img_height)

        elif self.suffix == '.xml':
            # 获取每个object的box信息
            x1 = round(float(object_info["bndbox"]["xmin"]))
            y1 = round(float(object_info["bndbox"]["ymin"]))
            x2 = round(float(object_info["bndbox"]["xmax"]))
            y2 = round(float(object_info["bndbox"]["ymax"]))

        elif self.suffix == '.json':
            # 获取每个object的box信息
            x1 = round(float(object_info["points"][0]))
            y1 = round(float(object_info["points"][1]))
            x2 = round(float(object_info["points"][2]))
            y2 = round(float(object_info["points"][3]))

        return x1, y1, x2, y2


    def verify_coordinates(self):
        # 如果有object，检查object是否合法
        for object_info in self.objects:
            # 如果点的个数不全，那么则不进行具体的坐标检查
            if self.verify_num_coordinates_for_object(object_info):
                # 坐标标准化
                x1, y1, x2, y2 = self.normalize_coordinates_for_object(object_info)
                
                # 检查每个object的坐标是否有问题
                self.verify_coordinates_for_object(x1, y1, x2, y2)


    def verify_metadata(self):
        # 提升作用域
        filename = None
        width = None
        height = None
        depth = None
        shape_type = None
        imagedata = None

        if self.suffix == '.txt':
            # 获取当前objects的所有类别索引，并计算最大值
            class_index = [int(object_info.split(' ')[0]) for object_info in self.objects]
            max_class_index = max(class_index)

        elif self.suffix == '.xml':
            # 获取的信息
            filename = self.xml_data["filename"]
            width = int(self.xml_data["size"]["width"])
            height = int(self.xml_data["size"]["height"])
            depth = int(self.xml_data["size"]["depth"])

            # 获取最大的类别索引（先判断每个object的类别是否在classes_dict中，如果不在则将这个类别保留（方便报错））
            class_index = [self.classes_dict.get(object_info['name'], object_info['name']) for object_info in self.objects]

            # 对class_index这个list进行遍历，看看有没有字符串，如果有则报错
            for cn in class_index:  # class name
                if isinstance(cn, str):
                    self.log(f"[{self.count}] Unknown class name found: {cn}.")
                    class_index.remove(cn)  # 把这个字符串删掉
            max_class_index = max(class_index) if class_index else 0

        elif self.suffix == '.json':

            # 获取信息
            filename = self.json_data['imagePath']
            imagedata = self.json_data['imageData']
            width = self.json_data['imageWidth']
            height = self.json_data['imageHeight']

            for object_info in self.objects:
                if object_info.get('shape_type', None) != 'rectangle':  # 'rectangle'
                    self.log(f"[{self.count}] The 'shape_type' is '{object_info.get('shape_type', None)}'(label) instead of 'rectangle'(image).")

            # 获取最大的类别索引（先判断每个object的类别是否在classes_dict中，如果不在则将这个类别保留（方便报错））
            class_index = [self.classes_dict.get(object_info['label'], object_info['label']) for object_info in self.objects]

            # 对class_index这个list进行遍历，看看有没有字符串，如果有则报错
            for cn in class_index:  # class name
                if isinstance(cn, str):
                    self.log(f"[{self.count}] Unknown class name found: {cn}.")
                    class_index.remove(cn)  # 把这个字符串删掉
            max_class_index = max(class_index) if class_index else 0

        # 检查信息
        if max_class_index > len(self.classes_dict) - 1:
            self.log(f"[{self.count}] The max class index of object ({max_class_index}) is out of boundary ({len(self.classes_dict) - 1}).")
        if filename and str(self.image.name) not in filename:
            self.log(f"[{self.count}] The filename is {filename}(label) instead of {str(self.image.name)}(image).")
        elif width and width != self.img_width:
            self.log(f"[{self.count}] The width is {width}(label) instead of {self.img_width}(image).")
        elif height and height != self.img_height:
            self.log(f"[{self.count}] The height is {height}(label) instead of {self.img_height}(image).")
        elif depth and depth != self.img_channel:
            self.log(f"[{self.count}] The channel is {depth}(label) instead of {self.img_channel}(image).")
        elif imagedata:
            self.log(f"[{self.count}] The 'imageData' is not empty.")
    
    
    def label_exists(self) -> bool:
        return self.label.exists()
    

    def start_and_receive_results(self) -> list:
        # 根据标签后缀读取标签
        if self.suffix == '.txt':
            self.objects = read_txt(self.label)
        elif self.suffix == '.xml':
            self.xml_data = read_xml(self.label)
            self.objects = self.xml_data.get("object", None)
        elif self.suffix == '.json':
            self.json_data = read_json(self.label)
            self.objects = self.json_data['shapes']
        else:
            raise NotImplementedError(
                f"❌ The current script only supports label with {colorstr('.txt, .json, .xml')}, "
                f"and does not support {colorstr(self.suffix)}!"
            )

        # 检查1：检查坐标点是存在
        if not self.objects:  # 坐标点不存在
            self.log(f"[{self.count}] There is no any objects.")
        else:  # 坐标点存在，进一步检查坐标是否正确
            # 检查2：坐标是否正确（包括点的个数是否=4）
            self.verify_coordinates()

            # 检查3：metadata是否正确
            self.verify_metadata()

        # 输出结果
        return self.msgs
    

def max_length_in_iterable(iterable):
    max_length = 0
    for item in iterable:
        try:
            # Try to get the length of the item
            length = len(item)
        except TypeError:
            # If item does not have length (e.g., an integer), skip
            length = 0
        # Update max_length if the current item's length is greater
        if length > max_length:
            max_length = length
    return max_length


def show_args(*args, **kwargs) -> PrettyTable:
    ptab = PrettyTable(field_names=['Arguments', 'Value'])
    ptab.align = 'l'
    interval_num = 5
    
    for v in args:
        if isinstance(v, argparse.Namespace):
            for _k, _v in vars(v).items():
                if _k in ('classes', 'class', 'cls'):
                    if _v:
                        ptab.add_row(['🌟 The number of classes', len(_v)])
                        for group_idx, i in enumerate(range(0, len(_v), interval_num), start=1):
                            # 检查是否到达列表末尾
                            if i + interval_num <= len(_v):
                                three_elements = _v[i:i+interval_num]
                            else:
                                three_elements = _v[i:]
                            ptab.add_row([f"{_k} [part{group_idx}({i + 1}~{i + len(three_elements)})]", three_elements])
                    else:  # args.classes == None
                        ptab.add_row([_k, _v])
                elif isinstance(_v, (list, tuple, set)):  # 解决长度问题
                    max_length = max_length_in_iterable(_v)
                    if max_length > 20:
                        for i, elem in enumerate(_v, start=1):
                            ptab.add_row([f"{_k}-{i}", elem])
                    else:
                        ptab.add_row([_k, _v])
                else:
                    ptab.add_row([_k, _v])
        else:
            ptab.add_row(['', v])

    for k, v in kwargs.items():
        if isinstance(v, argparse.Namespace):
            for _k, _v in vars(v).items():
                ptab.add_row([_k, _v])
        else:
            ptab.add_row([k, v])
    return ptab


