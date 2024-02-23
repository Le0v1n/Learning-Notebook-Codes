"""
+ 脚本说明：目标检测中xml标注文件转换为yolo格式
+ 用途：xml2yolo
+ 要求：要有对应的图片文件，这样读取到的尺寸信息是最准确的。
+ 注意：会生成日志文件
"""
import os
import tqdm
import datetime
import logging
from lxml import etree
from PIL import Image


"""============================ 需要修改的地方 ==================================="""
IMAGE_PATH = 'EXAMPLE_FOLDER/images'  # 图片路径
XML_PATH = 'EXAMPLE_FOLDER/annotations-xml'  # xml标签路径
LABELS_PATH = "EXAMPLE_FOLDER/labels"  # yolo标签保存路径
LOGGING_SAVE_FOLDERNAME = 'local-log'  # 日志的保存文件夹名称(不是路径，只是文件夹名称)

IMAGE_TYPE = '.jpg'  # 图片的格式

# 标签从0开始
class_dict = {"cat": 0, 
              "dog": 1}

DECIMAL_PLACES = 6  # 标签保留几位小数, 默认为6
OVERRIDE = True  # 是否要覆盖已存在txt文件
CLIP_OUT_OF_BOUNDARY = True  # 对于 xmin 或 ymin < 0 的情况，直接截断为 0; 对于 xmax 或 ymax > 图片尺寸的情况，直接截断图片最大尺寸
VERBOSE = False  # 终端不会打印日志了（日志仍会生成）
"""==============================================================================="""

"---------------------------------------日志---------------------------------------"
# 获取当前时间
current_time = datetime.datetime.now()  
formatted_time = current_time.strftime("%Y%m%d%H%M%S")  # 格式化为指定格式

script_path = os.path.abspath(__file__)  # 获取当前脚本的绝对路径
script_name = os.path.splitext(os.path.basename(script_path))[0]
log_save_folder = os.path.join(os.path.dirname(script_path), LOGGING_SAVE_FOLDERNAME)  # 获取当前脚本所在的文件夹名
log_filename = os.path.join(formatted_time + '-' + script_name + '.log')   # 获取文件夹名并拼接日志文件名
log_file_path = os.path.join(log_save_folder, log_filename)  # 拼接日志文件的完整路径
"---------------------------------------------------------------------------------"

# 读取所有的 xml 文件
xml_files = [file for file in os.listdir(XML_PATH) if file.lower().endswith('.xml')]
image_files = [file for file in os.listdir(IMAGE_PATH) if file.lower().endswith(IMAGE_TYPE)]
not_wanna_image_files = [file for file in os.listdir(IMAGE_PATH) if not file.lower().endswith(IMAGE_TYPE)]

"------------计数------------"
TOTAL_NUM = len(xml_files)  # 所有 xml 文件的数量
IMAGE_NUM = len(image_files)
NOT_WANNA_IMAGES_NUM = len(not_wanna_image_files)
SUCCEED_NUM = 0  # 成功转换为 yolo 格式的 xml 文件数量
SKIP_NUM = 0  # 跳过转换的 xml 文件数量
OBJ_NUM = 0  # 目标总数
NEG_NUM = 0  # 负样本数量
WARNING_NUM = 0  # 警告数量
ERROR_NUM = 0  # 错误数量
ERROR_LOGGER = dict()  # 保存错误信息的字典
ERROR_POINT_NUM = 0  # 坏点数量
"---------------------------"

# 释放资源
del image_files
del not_wanna_image_files

_str = (f" 图片路径: \033[1;33m{IMAGE_PATH}\033[0m"
        f"\n xml路径: \033[1;33m{XML_PATH}\033[0m"
        f"\n txt保存路径: \033[1;33m{LABELS_PATH}\033[0m"
        f"\n {IMAGE_TYPE}图片数量: \033[1;33m{IMAGE_NUM}\033[0m"
        f"\n 💡 不是 {IMAGE_TYPE} 图片数量: \033[1;33m{NOT_WANNA_IMAGES_NUM}\033[0m"
        f"\n 需要转换的xml文件数量: \033[1;33m{TOTAL_NUM}\033[0m"
        f"\n 日志保存路径: \033[1;33m{log_file_path}\033[0m"
        f"\n\t💡 是否要覆盖: \033[1;33m{OVERRIDE}\033[0m"
        f"\n\t💡 是否对越界坐标进行截断: \033[1;33m{CLIP_OUT_OF_BOUNDARY}\033[0m"
        f"\n\t💡 日志是否在终端显示(不影响日志保存): \033[1;33m{VERBOSE}\033[0m"
        f"\n\n请输入 \033[1;31m'yes'\033[0m 继续，输入其他停止"
        )

print(_str)
    
_INPUT = input()
if _INPUT != "yes":
    exit()

# 创建文件夹
if not os.path.exists(log_save_folder):
    os.mkdir(log_save_folder)
    
if not os.path.exists(LABELS_PATH):
    os.makedirs(LABELS_PATH, exist_ok=True)

# 配置日志输出的格式和级别
logging.basicConfig(filename=log_file_path, 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 创建日志记录器
logger = logging.getLogger()
logger.info(_str)

if VERBOSE:
    # 创建控制台处理器并添加到日志记录器
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)


def recalc_coordinate(xmin, ymin, xmax, ymax, json_path, WARNING_NUM=0, logger=None):
    if xmin > xmax or ymin > ymax:  # 出现错误
        WARNING_NUM += 1
        _xmin, _ymin, _xmax, _ymax = xmin, ymin, xmax, ymax  # 保留一下之前的坐标

        # 重新计算中心点坐标
        xcenter, ycenter = (xmin + xmax) / 2, (ymin + ymax) / 2
        
        # 根据中心点坐标(xcenter, ycenter)重新计算左上角坐标(xmin, ymin)和右上角坐标(xmax, ymax)
        width, height = abs(xmax - xmin), abs(ymax - ymin)
        
        # 计算和更新坐标
        xmin, ymin = xcenter - width / 2, ycenter - height / 2
        xmax, ymax = xcenter + width / 2, ycenter + height / 2

        logger.warning(f"坐标错误, 已重新计算!\n\t"
                       f"[[{_xmin}, {_ymin}], [{_xmax}, {_ymax}]] -> \n\t[[{xmin}, {ymin}], [{xmax}, {ymax}]]:\n\t"
                       f"{json_path}") if logger else ...

    return xmin, ymin, xmax, ymax, WARNING_NUM
        
        
def check_coordinate_validity(xmin, ymin, xmax, ymax, 
                              img_width, img_height,
                              CLIP_OUT_OF_BOUNDARY=False, ERROR_NUM=0, logger=None):
    """
    判断坐标是否合法（非负且xmin <= xmax，ymin <= ymax）。

    Parameters:
        xmin (float): 左上角 x 坐标
        ymin (float): 左上角 y 坐标
        xmax (float): 右下角 x 坐标
        ymax (float): 右下角 y 坐标
        
        CLIP_OUT_OF_BOUNDARY (bool): 是否对不合法的坐标进行截断修复

    Returns:
        bool: 如果坐标合法返回 True，否则返回 False
    """
    if xmin < 0: 
        _xmin = xmin
        if CLIP_OUT_OF_BOUNDARY:
            xmin = 0.0
        logger.error(msg=f"[xmin({_xmin}) < 0]\n\tNow xmin={xmin}\n\t{json_path}")
        ERROR_NUM += 1
    if ymin < 0: 
        _ymin = ymin
        if CLIP_OUT_OF_BOUNDARY:
            ymin = 0.0
        logger.error(msg=f"[ymin({_ymin}) < 0]\n\tNow ymin={ymin}\n\t{json_path}")
        ERROR_NUM += 1
    if xmax < 0: 
        _xmax = xmax
        if CLIP_OUT_OF_BOUNDARY:
            xmax = 0.0
        logger.error(msg=f"[xmax({_xmax}) < 0]\n\tNow xmax={xmax}\n\t{json_path}")
        ERROR_NUM += 1
    if ymax < 0: 
        _ymax = ymax
        if CLIP_OUT_OF_BOUNDARY:
            ymax = 0.0
        logger.error(msg=f"[ymax({_ymax}) < 0]\n\tNow ymax={ymax}\n\t{json_path}")
        ERROR_NUM += 1

    # 如果出现越界
    if xmin > img_width: 
        _xmin = xmin
        if CLIP_OUT_OF_BOUNDARY:
            xmin = float(img_width)
        logger.error(msg=f"[xmin > img_width({_xmin} > {img_width})]\n\tNow xmin={xmin}\n\t{json_path}")
        ERROR_NUM += 1
    if ymin > img_height: 
        _ymin = ymin
        if CLIP_OUT_OF_BOUNDARY:
            ymin = float(img_height)
        logger.error(msg=f"[ymin > img_height({_ymin} > {img_height})]\n\tNow ymin={ymin}\n\t{json_path}")
        ERROR_NUM += 1
    if xmax > img_width: 
        _xmax = xmax
        if CLIP_OUT_OF_BOUNDARY:
            xmax = float(img_width)
        logger.error(msg=f"[xmax > img_width({_xmax} > {img_width})]\n\tNow xmax={xmax}\n\t{json_path}")
        ERROR_NUM += 1
    if ymax > img_height: 
        _ymax = ymax
        if CLIP_OUT_OF_BOUNDARY:
            ymax = float(img_height)
        logger.error(msg=f"[ymax > img_height({_ymax} > {img_height})]\n\tNow ymax={ymax}\n\t{json_path}")
        ERROR_NUM += 1
    
    return xmin, ymin, xmax, ymax, ERROR_NUM
    
    
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


process_bar = tqdm.tqdm(total=TOTAL_NUM, desc="xml2yolo", unit='xml')
for xml_name in xml_files:
    process_bar.set_description(f"\033[1;31m{xml_name}\033[0m")
    xml_pre, xml_ext = os.path.splitext(xml_name)  # 分离文件名和后缀
    xml_path = os.path.join(XML_PATH, xml_name)  # xml文件完整路径

    # 打开xml文件
    with open(xml_path) as fid:
        xml_str = fid.read()
        
    # 将XML字符串编码为字节序列
    xml_bytes = xml_str.encode('utf-8')

    # 使用lxml解析字节序列的XML数据
    xml = etree.fromstring(xml_bytes)
    data = parse_xml_to_dict(xml)["annotation"]
    
    # 构建图片路径
    img_full_path = os.path.join(IMAGE_PATH, xml_pre) + IMAGE_TYPE
    
    if os.path.exists(img_full_path):
        img = Image.open(img_full_path)
        img_width, img_height = img.size
        img.close()
    else:  # 图片不存在
        WARNING_NUM += 1
        logger.warning(msg=f"[图片不存在, 使用xml中的尺寸信息!\n\t{xml_path}")
        img_width = int(data["size"]["width"])  # 图片宽度
        img_height = int(data["size"]["height"])  # 图片高度
    
    txt_path = os.path.join(LABELS_PATH, xml_pre + ".txt")
    with open(txt_path, "w") as f:
        # 如果没有object -> 负样本
        objects = data.get("object")
        if objects is None:
            NEG_NUM += 1 
            SUCCEED_NUM += 1
            process_bar.update()
            continue

        for index, obj in enumerate(data["object"]):
            # 检查是否有缺点的情况
            _pt_num = len(obj["bndbox"])
            
            if _pt_num != 4:
                logger.error(msg=f"[坏点 = {_pt_num}]\n\t{xml_path}")
                ERROR_POINT_NUM += 1
                continue
            
            # 获取每个object的box信息
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            
            # 检查坐标点是否①负数；②越界
            xmin, ymin, xmax, ymax, ERROR_NUM = check_coordinate_validity(xmin, ymin, xmax, ymax, 
                                                                          img_width, img_height,
                                                                          CLIP_OUT_OF_BOUNDARY, ERROR_NUM, 
                                                                          logger=logger)
            
            # 检查是否需要重新计算坐标
            xmin, ymin, xmax, ymax, WARNING_NUM = recalc_coordinate(xmin, ymin, xmax, ymax, 
                                                                    xml_path, WARNING_NUM, logger=logger)
            
            class_name = obj["name"]
            class_index = class_dict[class_name]

            # 将box信息转换到yolo格式
            xcenter = xmin + (xmax - xmin) / 2  # 中心点的x
            ycenter = ymin + (ymax - ymin) / 2  # 中心点的y
            w = xmax - xmin  # 宽度
            h = ymax - ymin  # 高度

            # 绝对坐标转相对坐标，保存DECIMAL_PLACES位小数
            xcenter = round(xcenter / img_width, DECIMAL_PLACES)
            ycenter = round(ycenter / img_height, DECIMAL_PLACES)
            w = round(w / img_width, DECIMAL_PLACES)
            h = round(h / img_height, DECIMAL_PLACES)
            
            # 要输入txt文本的内容
            info = [str(i) for i in [class_index, xcenter, ycenter, w, h]]  # c, x, y, w, h

            # 写入txt
            if index == 0:
                f.write(" ".join(info))
            else:
                f.write("\n" + " ".join(info))
    SUCCEED_NUM += 1
    process_bar.update()
process_bar.close()

_str = (f"xml2yolo已完成，详情如下：\n\t"
        f"👌成功: {SUCCEED_NUM}/{TOTAL_NUM}\n\t"
        f"👌跳过: {SKIP_NUM}/{TOTAL_NUM}\n\t"
        f"Object数量: {OBJ_NUM}\n\t"
        f"每张图片平均Object数量: {OBJ_NUM/TOTAL_NUM:.2f}\n\t"
        f"坏点的数量为: {ERROR_POINT_NUM}\n\n"
        f"结果保存在: {LABELS_PATH}"
        f"日志保存在: {log_file_path}")

logger.info(_str)
print(_str) if not VERBOSE else ...

if SUCCEED_NUM + SKIP_NUM == TOTAL_NUM:
    _str = (f"👌 No Problems in data numbers")
    logger.info(_str)
    print(_str) if not VERBOSE else ...