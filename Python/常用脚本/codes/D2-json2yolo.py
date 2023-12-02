"""
+ 脚本说明：目标检测中json标注文件转换为yolo格式
+ 用途：json2yolo
+ 要求：要有对应的图片文件，这样读取到的尺寸信息是最准确的。
+ 注意: 该脚本会生成日志文件!
"""
import os
import cv2
import json
import numpy as np
import tqdm
import logging
import os
import datetime


"""============================ 需要修改的地方 ==================================="""
IMAGE_PATH = 'EXAMPLE_FOLDER/images'  # 图片路径
JSON_PATH = 'EXAMPLE_FOLDER/annotations-json'  # json标签路径
TXT_SAVE_PATH = "EXAMPLE_FOLDER/labels"  # yolo标签保存路径

IMAGE_TYPE = '.jpg'

# 标签字典
classes_dict = {'cat': 0,
              'dog': 1,
              }

OVERRIDE = False  # 是否要覆盖已存在txt文件
CLIP_OUT_OF_BOUNDARY = True  # 对于 xmin 或 ymin < 0 的情况，直接截断为 0; 对于 xmax 或 ymax > 图片尺寸的情况，直接截断图片最大尺寸
EXCHANGE_COORDINATES = True  # 是否允许交换坐标(推荐打开)
"""==============================================================================="""

"---------------------------------------日志---------------------------------------"
# 获取当前时间
current_time = datetime.datetime.now()  
formatted_time = current_time.strftime("%Y%m%d%H%M%S")  # 格式化为指定格式

script_path = os.path.abspath(__file__)  # 获取当前脚本的绝对路径
script_folder = os.path.dirname(script_path)  # 获取当前脚本所在的文件夹名
script_name = os.path.splitext(os.path.basename(script_path))[0]
log_filename = 'local_log-' + formatted_time + '-' + script_name + '.log'  # 获取文件夹名并拼接日志文件名
log_file_path = os.path.join(script_folder, log_filename)  # 拼接日志文件的完整路径

# 配置日志输出的格式和级别
logging.basicConfig(filename=log_file_path, 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 创建日志记录器
logger = logging.getLogger()

# 创建控制台处理器并添加到日志记录器
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)
"---------------------------------------------------------------------------------"

# 读取所有 .json 文件
annotation_list = [file for file in os.listdir(JSON_PATH) if file.endswith('.json')]
image_list = [file for file in os.listdir(IMAGE_PATH) if file.endswith(IMAGE_TYPE)]

"------------计数------------"
TOTAL_NUM = len(annotation_list)
SUCCEED_NUM = 0
SKIP_NUM = 0
OBJ_NUM = 0
ERROR_NUM = 0
ERROR_LOGGER = dict()
"---------------------------"

logger.info(f" 图片路径: \033[1;33m{IMAGE_PATH}\033[0m"
            f"\n json路径: \033[1;33m{JSON_PATH}\033[0m"
            f"\n txt保存路径: \033[1;33m{TXT_SAVE_PATH}\033[0m"
            f"\n 图片数量: \033[1;33m{len(image_list)}\033[0m"
            f"\n 需要转换的json文件数量: \033[1;33m{TOTAL_NUM}\033[0m"
            f"\n\t💡 是否要覆盖: \033[1;33m{OVERRIDE}\033[0m"
            f"\n\t💡 是否对越界坐标进行截断: \033[1;33m{CLIP_OUT_OF_BOUNDARY}\033[0m"
            f"\n\t💡 是否交换两个坐标: \033[1;33m{EXCHANGE_COORDINATES}\033[0m"
            f"\n\n请输入 \033[1;31m'yes'\033[0m 继续，输入其他停止"
            )
_INPUT = input()
if _INPUT != "yes":
    exit()

if not os.path.exists(TXT_SAVE_PATH):
    os.mkdir(TXT_SAVE_PATH)


# 创建进度条
progress_bar = tqdm.tqdm(total=TOTAL_NUM, desc="json -> yolo(.txt)", unit=" .txt")
for _, json_name in enumerate(annotation_list):
    progress_bar.set_description(f"Convert \033[1;31m{json_name}\033[0m")
    pre, ext = os.path.splitext(json_name)  # 分离前缀和后缀
    json_path = os.path.join(JSON_PATH, json_name)  # json文件完整路径
    img_path = os.path.join(IMAGE_PATH, pre) + IMAGE_TYPE  # 最终保存的图片文件完整路径
    txt_path = os.path.join(TXT_SAVE_PATH, pre) + '.txt'  # 最终保存的txt文件完整路径

    # 如果目标txt文件存在则跳过
    if not OVERRIDE and os.path.exists(txt_path):
        SKIP_NUM += 1
        progress_bar.update()
        continue
    
    # 目标txt文件不存在 -> 创建txt文件
    with open(json_path, 'r') as fr:  # 打开json文件
        result = json.load(fr)

    # 读取图片的宽高信息
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[0:2]
    
    # 获取所有 object 的信息
    all_object_info = result['shapes']
    
    # 创建 txt 文件并写入内容
    with open(txt_path, 'w') as target_file:
        for idx, obj_info in enumerate(all_object_info):
            label = classes_dict[obj_info['label']]  # 读取当前obj的类别
            points = np.array(obj_info['points'])  # 读取当前obj的位置
            xmin, ymin, xmax, ymax = points[0][0], points[0][1], points[1][0], points[1][1]
            
            """
                有些标注工具没有那么智能，反着标注的数据不能智能调整左上角坐标和右下角坐标，因此会导致在转换后
                出现负数的情况，因此我们可以加一个简单的逻辑判断从而规避这种问题
            """
            # 如果出现负数
            if xmin < 0: 
                ERROR_NUM += 1
                if CLIP_OUT_OF_BOUNDARY:
                    xmin = 0.0
                logger.error(msg=f"[xmin({xmin}) < 0] in {json_path}")
            if ymin < 0: 
                if CLIP_OUT_OF_BOUNDARY:
                    ymin = 0.0
                logger.error(msg=f"[ymin({ymin}) < 0] in {json_path}")
                ERROR_NUM += 1
            if xmax < 0: 
                if CLIP_OUT_OF_BOUNDARY:
                    xmax = 0.0
                logger.error(msg=f"[xmax({xmax}) < 0] in {json_path}")
                ERROR_NUM += 1
            if ymax < 0: 
                if CLIP_OUT_OF_BOUNDARY:
                    ymax = 0.0
                logger.error(msg=f"[ymax({ymax}) < 0] in {json_path}")
                ERROR_NUM += 1

            # 如果出现越界
            if xmin > img_width: 
                if CLIP_OUT_OF_BOUNDARY:
                    xmin = float(img_width)
                logger.error(msg=f"[xmin > img_width({xmin} > {img_width})] in {json_path}")
                ERROR_NUM += 1
            if ymin > img_height: 
                if CLIP_OUT_OF_BOUNDARY:
                    ymin = float(img_height)
                logger.error(msg=f"[ymin > img_height({ymin} > {img_height})] in {json_path}")
                ERROR_NUM += 1
            if xmax > img_width: 
                if CLIP_OUT_OF_BOUNDARY:
                    xmax = float(img_width)
                logger.error(msg=f"[xmax > img_width({xmax} > {img_width})] in {json_path}")
                ERROR_NUM += 1
            if ymax > img_height: 
                if CLIP_OUT_OF_BOUNDARY:
                    ymax = float(img_height)
                logger.error(msg=f"[ymax > img_height({ymax} > {img_height})] in {json_path}")
                ERROR_NUM += 1
            
            # 当出现 xmin > xmax 或 ymin > ymax 时 -> 交换坐标
            if EXCHANGE_COORDINATES and (xmin > xmax or ymin > ymax):
                xmin, ymin, xmax, ymax = xmax, ymax, xmin, ymin  # 交换坐标
                logger.warning(f"两个坐标点反了，已交换: {json_path}")

            # 计算YOLO格式的坐标
            x_center = xmin + (xmax - xmin) / 2
            y_center = ymin + (ymax - ymin) / 2
            w = (xmax - xmin)  # 不需要 / 2 嗷
            h = (ymax - ymin)  # 不需要 / 2 嗷
            
            # 绝对坐标转相对坐标，保存6位小数
            x_center = round(x_center / img_width, 6)
            y_center = round(y_center / img_height, 6)
            w = round(w / img_width, 6)
            h = round(h / img_height, 6)
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

print(f"json2yolo已完成，详情如下：\n\t"
      f"👌成功: {SUCCEED_NUM}/{TOTAL_NUM}\n\t"
      f"👌跳过: {SKIP_NUM}/{TOTAL_NUM}\n\t"
      f"生成的Object数量: {OBJ_NUM}")

if SUCCEED_NUM + SKIP_NUM == TOTAL_NUM:
    print(f"👌 No Problems")