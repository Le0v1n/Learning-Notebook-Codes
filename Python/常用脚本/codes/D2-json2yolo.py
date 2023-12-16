"""
    json转yolo的txt
"""
import os
import cv2
import json
import numpy as np
import tqdm
import logging
import os
import datetime
from tabulate import tabulate


"""============================ 需要修改的地方 ==================================="""
IMAGE_PATH = 'EXAMPLE_FOLDER/images'
ANNOTATION_PATH = 'EXAMPLE_FOLDER/annotations'
LABELS_PATH = 'EXAMPLE_FOLDER/labels'

IMAGE_TYPE = '.jpg'  # 图片类型

# 选择任务确定标签字典
class_dict = {"cat": 0, 
              "dog": 1}

CONVERT_CLASS_DICT = False  # 是否需要把类别字典的key和value翻转一下?

# 是否为负样本创建txt文件
CREATE_TXT_4_NEG = True

DECIMAL_PLACES = 6  # 标签保留几位小数, 默认为6
OVERRIDE = True  # 是否要覆盖已存在txt文件
CLIP_OUT_OF_BOUNDARY = True  # 对于 xmin 或 ymin < 0 的情况，直接截断为 0; 对于 xmax 或 ymax > 图片尺寸的情况，直接截断图片最大尺寸
VERBOSE = False  # 终端不会打印日志了（日志仍会生成）
"""==============================================================================="""

label_dict = {v: k for k, v in label_dict.items()}  # 翻转字典

"---------------------------------------日志---------------------------------------"
script_path = os.path.abspath(__file__)  # 获取当前脚本的绝对路径
script_name = os.path.splitext(os.path.basename(script_path))[0]  # 当前脚本的名称(没有.py后缀)
script_folder_path = os.path.dirname(script_path)  # 获取当前脚本所在的文件夹名
log_folder_path = os.path.join(script_folder_path, LOG_FOLDER_NAME)  # 存放log的文件夹路径

formatted_time = datetime.datetime.now().strftime("%Y%m%d-%H_%M_%S")  # 获取当前时间并格式化为指定格式
log_filename = os.path.join(log_folder_path, formatted_time + '-' + script_name + '.log')   # 获取文件夹名并拼接日志文件名
log_file_path = os.path.join(script_folder_path, log_filename)  # 拼接日志文件的完整路径
"---------------------------------------------------------------------------------"

# 读取所有 .json 文件
annotation_list = [file for file in os.listdir(ANNOTATION_PATH) if file.endswith('.json')]
image_list = [file for file in os.listdir(IMAGE_PATH) if file.endswith(IMAGE_TYPE)]
not_image_list = [file for file in os.listdir(IMAGE_PATH) if not file.endswith(IMAGE_TYPE)]

"------------计数------------"
NOT_WANNA_IMAGES_NUM = len(not_image_list)
TOTAL_NUM = len(annotation_list)
SUCCEED_NUM = 0
SKIP_NUM = 0
OBJ_NUM = 0
WARNING_NUM = 0
ERROR_NUM = 0
ERROR_LOGGER = dict()
ERROR_POINT_NUM = 0
NEG_NUM = 0
"---------------------------"
del not_image_list

_str = [
    ["图片路径", IMAGE_PATH],
    ["json路径", ANNOTATION_PATH],
    ["txt保存路径", LABELS_PATH],
    ["", ""],
    [f"{IMAGE_TYPE}图片数量", len(image_list)],
    ["💡 不是图片数量", NOT_WANNA_IMAGES_NUM],
    ["需要转换的json文件数量", TOTAL_NUM],
    ["", ""],
    ["💡 是否为负样本创建txt文件", CREATE_TXT_4_NEG],
    ["💡 是否要覆盖", OVERRIDE],
    ["💡 是否对越界坐标进行截断", CLIP_OUT_OF_BOUNDARY],
    ["", ""],
    ["日志保存路径", log_file_path],
    ["💡 日志是否在终端显示", VERBOSE],
]

# 添加类别字典
_str.append([])
for count, (key, value) in enumerate(label_dict.items()):
    if key.isdigit():  # 如果是数字
        _str.append([f"类别-{count}", value])
    elif value.isdigit():
        _str.append([f"类别-{count}", key])
    else:  # 类别字典有问题，直接kv显示
        _str.append([key, value])

_str = tabulate(_str, headers=["PARAMs", "VALUE"], tablefmt="outline")
print(f"{_str}\n\n"
      f"请输入 'yes' 继续，输入其他停止")
    
_INPUT = input()
if _INPUT != "yes":
    exit()

# 配置日志输出的格式和级别
os.mkdir(log_folder_path) if not os.path.exists(log_folder_path) else ...
logging.basicConfig(filename=log_file_path, 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 创建日志记录器
logger = logging.getLogger()
logger.info(f"\n{_str}")

if VERBOSE:
    # 创建控制台处理器并添加到日志记录器
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

if not os.path.exists(LABELS_PATH):
    os.mkdir(LABELS_PATH)
    
    
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

        logger.warning(f"坐标错误, 已重新计算!\n\t\t"
                       f"[({_xmin}, {_ymin}), ({_xmax}, {_ymax})]\n"
                       f"---new--->\n\t\t"
                       f"[({xmin}, {ymin}), ({xmax}, {ymax})]:\n\t"
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
        logger.error(msg=f"[xmin({_xmin}) < 0] ---new---> xmin={xmin}\n\t{json_path}")
        ERROR_NUM += 1
    if ymin < 0: 
        _ymin = ymin
        if CLIP_OUT_OF_BOUNDARY:
            ymin = 0.0
        logger.error(msg=f"[ymin({_ymin}) < 0] ---new---> ymin={ymin}\n\t{json_path}")
        ERROR_NUM += 1
    if xmax < 0: 
        _xmax = xmax
        if CLIP_OUT_OF_BOUNDARY:
            xmax = 0.0
        logger.error(msg=f"[xmax({_xmax}) < 0] ---new---> xmax={xmax}\n\t{json_path}")
        ERROR_NUM += 1
    if ymax < 0: 
        _ymax = ymax
        if CLIP_OUT_OF_BOUNDARY:
            ymax = 0.0
        logger.error(msg=f"[ymax({_ymax}) < 0] ---new---> ymax={ymax}\n\t{json_path}")
        ERROR_NUM += 1

    # 如果出现越界
    if xmin > img_width: 
        _xmin = xmin
        if CLIP_OUT_OF_BOUNDARY:
            xmin = float(img_width)
        logger.error(msg=f"[xmin > img_width({_xmin} > {img_width})] ---new---> xmin={xmin}\n\t{json_path}")
        ERROR_NUM += 1
    if ymin > img_height: 
        _ymin = ymin
        if CLIP_OUT_OF_BOUNDARY:
            ymin = float(img_height)
        logger.error(msg=f"[ymin > img_height({_ymin} > {img_height})] ---new---> ymin={ymin}\n\t{json_path}")
        ERROR_NUM += 1
    if xmax > img_width: 
        _xmax = xmax
        if CLIP_OUT_OF_BOUNDARY:
            xmax = float(img_width)
        logger.error(msg=f"[xmax > img_width({_xmax} > {img_width})] ---new---> xmax={xmax}\n\t{json_path}")
        ERROR_NUM += 1
    if ymax > img_height: 
        _ymax = ymax
        if CLIP_OUT_OF_BOUNDARY:
            ymax = float(img_height)
        logger.error(msg=f"[ymax > img_height({_ymax} > {img_height})] ---new---> ymax={ymax}\n\t{json_path}")
        ERROR_NUM += 1
    
    return xmin, ymin, xmax, ymax, ERROR_NUM


# 创建进度条
progress_bar = tqdm.tqdm(total=TOTAL_NUM, desc="json2yolo", unit=" .txt")
for _, json_name in enumerate(annotation_list):
    progress_bar.set_description(f"\033[1;31m{json_name}\033[0m")
    pre, ext = os.path.splitext(json_name)  # 分离前缀和后缀
    json_path = os.path.join(ANNOTATION_PATH, json_name)  # json文件完整路径
    img_path = os.path.join(IMAGE_PATH, pre) + IMAGE_TYPE  # 最终保存的图片文件完整路径
    txt_path = os.path.join(LABELS_PATH, pre) + '.txt'  # 最终保存的txt文件完整路径

    # 如果目标txt文件存在则跳过
    if not OVERRIDE and os.path.exists(txt_path):
        SKIP_NUM += 1
        progress_bar.update()
        continue
    
    # 目标txt文件不存在 -> 创建txt文件
    with open(json_path, 'r') as fr:  # 打开json文件
        result = json.load(fr)

    # 读取图片的宽高信息
    if not os.path.exists(img_path):  # 如果对应的图片不存在
        ERROR_NUM += 1
        logger.error(msg=f"[图片不存在!] {img_path}\n\t{json_path}")
        progress_bar.update()
        continue
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[0:2]  # 👌
    
    # 获取所有 object 的信息 -> e.g. [{'label': 'dough_hambuger', 'text': '', 'points': [(619.1836734693877, 347.14285714285717), (657.9591836734694, 377.7551020408164)], 'group_id': None, 'shape_type': 'rectangle', 'flags': {}}]
    all_object_info = result['shapes']
    
    # 创建 txt 文件并写入内容
    with open(txt_path, 'w') as target_file:
        for idx, obj_info in enumerate(all_object_info):
            label = label_dict[obj_info['label']]  # 读取当前obj的类别
            points = np.array(obj_info['points'])  # 读取当前obj的位置 -> e.g. [(619.18367347 347.14285714] [657.95918367 377.75510204]]

            # 防止出现一个点
            _pt_len = len(points)
            if _pt_len != 2:
                logger.error(msg=f"[坏点 = {_pt_len}]\n\t{json_path}")
                ERROR_POINT_NUM += 1
                continue

            xmin, ymin, xmax, ymax = points[0][0], points[0][1], points[1][0], points[1][1]
            """
                points[0]: [619.18367347 347.14285714]
                points[1]: [657.95918367 377.75510204]
            """
            
            """
                有些标注工具没有那么智能，反着标注的数据不能智能调整左上角坐标和右下角坐标，因此会导致在转换后
                出现负数的情况，因此我们可以加一个简单的逻辑判断从而规避这种问题
            """
            # 检查坐标点是否①负数；②越界
            xmin, ymin, xmax, ymax, ERROR_NUM = check_coordinate_validity(xmin, ymin, xmax, ymax, 
                                                                          img_width, img_height,
                                                                          CLIP_OUT_OF_BOUNDARY, ERROR_NUM, 
                                                                          logger=logger)
            
            # 检查是否需要重新计算坐标
            xmin, ymin, xmax, ymax, WARNING_NUM = recalc_coordinate(xmin, ymin, xmax, ymax, 
                                                                    json_path, WARNING_NUM, logger=logger)

            # 计算YOLO格式的坐标
            x_center = xmin + (xmax - xmin) / 2
            y_center = ymin + (ymax - ymin) / 2
            w = (xmax - xmin)  # 不需要 / 2 嗷
            h = (ymax - ymin)  # 不需要 / 2 嗷
            
            # 绝对坐标转相对坐标，保存6位小数
            x_center = round(x_center / img_width, DECIMAL_PLACES)
            y_center = round(y_center / img_height, DECIMAL_PLACES)
            w = round(w / img_width, DECIMAL_PLACES)
            h = round(h / img_height, DECIMAL_PLACES)
            obj_info = list(map(str, [label, x_center, y_center, w, h]))
            # target_file.write(obj_info+'\n')

            if idx == 0:
                target_file.write(" ".join(obj_info))
            else:
                target_file.write("\n" + " ".join(obj_info))

            OBJ_NUM += 1
    SUCCEED_NUM += 1
    progress_bar.update(1)
progress_bar.close()

# 最后考虑到可能有部分图片没有标签，所以我们主动给他们创建负样本
if CREATE_TXT_4_NEG:
    for image_name in tqdm.tqdm(image_list):
        pre, ext = os.path.splitext(image_name)
        txt_path = os.path.join(LABELS_PATH, pre) + '.txt'  # 最终保存的txt文件完整路径

        # 如果目标txt文件存在则跳过
        if os.path.exists(txt_path):
            continue
        
        # 如果目标 txt 文件不存在，创建空的 txt 文件
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            image_path = os.path.join(IMAGE_PATH, image_name)
            NEG_NUM += 1
            logger.warning(msg=f"[负样本] [{NEG_NUM}] {image_path}")

_str = [
    ["json2yolo", "已完成!"],
    ["成功转换数量", f"{SUCCEED_NUM}/{TOTAL_NUM}"],
    ["跳过转换数量", f"{SKIP_NUM}/{TOTAL_NUM}"],
    ["负样本数量", NEG_NUM],
    ["", ""],
    ["Object数量", OBJ_NUM],
    ["每张图片平均Object数量", f"{OBJ_NUM/TOTAL_NUM:.2f}"],
    ["坏点的数量为", ERROR_POINT_NUM],
    ["", ""],
    ["结果保存路径", LABELS_PATH],
    ["日志保存路径", log_file_path],
]

_str = tabulate(_str, headers=["PARAMs", "VALUE"], tablefmt="outline")

logger.info(f"\n{_str}")
print(_str) if not VERBOSE else ...

if SUCCEED_NUM + SKIP_NUM == TOTAL_NUM:
    _str = (f"👌 No Problems in data numbers")
    logger.info(_str)
    print(_str) if not VERBOSE else ...
