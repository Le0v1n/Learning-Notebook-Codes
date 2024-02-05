"""
+ 脚本说明：目标检测中yolo标注文件转换为json格式
+ 用途：YOLO 模型推理得到 txt 文件 -> 转换为 json 标注文件。
+ 要求：要有对应的图片文件，这样读取到的尺寸信息是最准确的。
"""
import os
import cv2
import tqdm
import json
import sys
sys.path.append('/mnt/f/Projects/本地代码/Learning-Notebook-Codes')
from Datasets.coco128.classes import coco128_class


"""============================ 需要修改的地方 ==================================="""
dataset_path = 'Datasets/coco128/train'  # 🧡数据集路径
classes_dict = coco128_class  # 🧡类别字典

image_folder_name = 'images'  # 图片文件夹名称
txt_folder_name = 'labels'  # txt文件夹名称
json_save_folder_path = 'annotations-json'  # json文件夹名称

image_type = '.jpg'  # 图片类型
create_empty_json_for_neg = True  # 是否为负样本生成对应的空的json文件
decimal_places = 6  # 标签保留几位小数, 默认为6

# 生成的 Json 文件基础信息
__version = "0.2.2"
__imageData = None
"""==============================================================================="""

# 组合路径
IMAGE_PATH = os.path.join(dataset_path, image_folder_name)
TXT_PATH = os.path.join(dataset_path, txt_folder_name)
JSON_PATH = os.path.join(dataset_path, json_save_folder_path)

txt_file_list = [file for file in os.listdir(TXT_PATH) if file.endswith("txt") and file != 'classes.txt']

"------------计数------------"
TOTAL_NUM = len(txt_file_list)
SUCCEED_NUM = 0  # 成功创建json数量
SKIP_NUM = 0  # 跳过创建json文件数量
OBJECT_NUM = 0  # object数量
ERROR_NUM = 0  # 没有对应图片
ERROR_LIST = []
"---------------------------"

_str = (f"💡 图片路径: \033[1;33m{IMAGE_PATH}\033[0m"
        f"\n💡 TXT文件路径为: \033[1;33m{TXT_PATH}\033[0m"
        f"\n💡 JSON文件路径为: \033[1;33m{JSON_PATH}\033[0m"
        f"\n 所有TXT文件数量: \033[1;33m{TOTAL_NUM}\033[0m"
        f"\n 类别字典为:")

for idx, value in classes_dict.items():
    _str += f"\n\t[{idx}] {value}"

_str += f"\n\n请输入 \033[1;31m'yes'\033[0m 继续，输入其他停止"
print(_str)

_INPUT = input()
if _INPUT != "yes":
    exit()

if not os.path.exists(JSON_PATH):
    os.makedirs(JSON_PATH)

process_bar = tqdm.tqdm(total=TOTAL_NUM, desc="yolo2json", unit='.txt')
for i, txt_name in enumerate(txt_file_list):
    process_bar.set_description(f"Process in \033[1;31m{txt_name}\033[0m")
    txt_pre, txt_ext = os.path.splitext(txt_name)  # 分离前缀和后缀

    # 完整路径
    txt_path = os.path.join(TXT_PATH, txt_name)
    image_path = os.path.join(IMAGE_PATH, txt_pre) + image_type
    json_save_path = os.path.join(JSON_PATH, txt_pre) + '.json'
        
    # 打开 txt 文件
    txtFile = open(txt_path)
    txtList = txtFile.readlines()  # 以一行的形式读取txt所有内容
    
    if not txtList and not create_empty_json_for_neg:  # 如果 txt 文件内容为空且不允许为负样本创建json文件
        SKIP_NUM += 1
        process_bar.update()
        continue
    
    # 如果图片不存在 -> 报错且跳过
    if not os.path.exists(image_path):  
        ERROR_NUM += 1
        ERROR_LIST.append(txt_path)
        process_bar.update()
        continue
    
    # 读取图片
    img = cv2.imread(image_path)
    height, width, channel = img.shape  # 获取图片尺寸
    
    # 创建 Json 文件的内容
    json_data = {"version": __version,
                 "flags": {},
                 "shapes": [],
                 "imagePath": f'../{os.path.join(os.path.basename(IMAGE_PATH), os.path.basename(image_path))}',  # 图片路径
                 "imageData": __imageData,
                 "imageHeight": height,
                 "imageWidth": width
                }
    
    # 读取 txt 内容，追加 json 文件的 shapes 内容
    for line in txtList:  # 正样本(txt内容不为空)
        # oneline: ['0', '0.660937', '0.161111', '0.0625', '0.107407'] -> [标签索引, x_center, y_center, w, h]
        oneline = line.strip().split(" ")
        
        # 获取坐标并转换为左上和右下的形式
        x_center, y_center, w, h = float(oneline[1]), float(oneline[2]), float(oneline[3]), float(oneline[4])
        
        # 将归一化的坐标还原为真实坐标
        x_center, y_center = x_center * width, y_center * height  # 还原中心点坐标
        w, h = w * width, h * height  # 还原宽度和高度
        
        xmin = round(x_center - w / 2, decimal_places)
        ymin = round(y_center - h / 2, decimal_places)
        xmax = round(x_center + w / 2, decimal_places)
        ymax = round(y_center + h / 2, decimal_places)
        
        # 添加到 shapes 列表中
        json_data["shapes"].append({
            "label": classes_dict[oneline[0]],
            "text": "",
            "points": [
                [xmin, ymin],
                [xmax, ymax]
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        })
        
        OBJECT_NUM += 1

    # 保存为json文件
    with open(json_save_path, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=2)
    
    SUCCEED_NUM += 1
    process_bar.update()
process_bar.close()

for idx, e in enumerate(ERROR_LIST):
    print("没有对应图片的TXT文件如下:") if idx == 0 else ...
    print(f"[{idx + 1}] {e}")

print(f"👌yolo2json已完成, 详情如下:"
      f"\n\t成功转换文件数量/总文件数量 = \033[1;32m{SUCCEED_NUM}\033[0m/{TOTAL_NUM}"
      f"\n\t跳过转换文件数量/总文件数量 = \033[1;31m{SKIP_NUM}\033[0m/{TOTAL_NUM}"
      f"\n\t所有样本的 object 数量/总文件数量 = \033[1;32m{OBJECT_NUM}\033[0m/{TOTAL_NUM}"
      f"\n\t平均每个json文件中object的数量为: {int(OBJECT_NUM / SUCCEED_NUM)}"
      f"\n\t没有对应图片的数量为: {ERROR_NUM}"
      f"\n\t结果保存路径为: {JSON_PATH}")

if SUCCEED_NUM + SKIP_NUM + ERROR_NUM == TOTAL_NUM:
    print(f"\n👌 \033[1;32mNo Problem\033[0m")
else:
    print(f"\n🤡 \033[1;31m貌似有点问题, 请仔细核查!\033[0m")