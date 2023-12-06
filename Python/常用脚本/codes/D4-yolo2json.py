"""
+ 脚本说明：目标检测中yolo标注文件转换为json格式
+ 用途：YOLO 模型推理得到 txt 文件 -> 转换为 json 标注文件。
+ 要求：要有对应的图片文件，这样读取到的尺寸信息是最准确的。
"""
import os
import cv2
import tqdm
import json


"""============================ 需要修改的地方 ==================================="""
IMAGE_PATH = "EXAMPLE_FOLDER/images"  # 原图文件夹路径
TXT_PATH = "EXAMPLE_FOLDER/labels"  # 原txt标签文件夹路径
JSON_PATH = "EXAMPLE_FOLDER/annotations-json"  # 保存json文件夹路径
IMAGE_TYPE = '.jpg'
create_empty_json_for_neg = True  # 是否为负样本生成对应的空的json文件

classes_dict = {
    '0': "cat",
    '1': 'dog'
}

# Json 文件基础信息
__version = "0.2.2"
__flags = {},
__imageData = None
"""==============================================================================="""

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
        f"\n\n请输入 \033[1;31m'yes'\033[0m 继续，输入其他停止")
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
    image_path = os.path.join(IMAGE_PATH, txt_pre) + IMAGE_TYPE
    json_save_path = os.path.join(JSON_PATH, txt_pre) + '.json'
        
    # 打开 txt 文件
    txtFile = open(txt_path)
    txtList = txtFile.readlines()  # 以一行的形式读取txt所有内容
    
    if not txtList and not create_empty_json_for_neg:  # 如果 txt 文件内容为空且不允许为负样本创建json文件
        SKIP_NUM += 1
        process_bar.update()
        continue
    
    # 读取图片
    if not os.path.exists(image_path):
        ERROR_NUM += 1
        ERROR_LIST.append(txt_path)
        process_bar.update()
        continue
    img = cv2.imread(image_path)
    height, width, channel = img.shape
    
    # 创建 Json 文件的内容
    json_data = {"version": __version,
               "flags": __flags,
               "shapes": [],
               "imagePath": f"{txt_pre + IMAGE_TYPE}",
               "imageData": __imageData,
               "imageHeight": height,
               "imageWidth": width
               }
    
    # 读取 txt 内容，追加 json 文件的 shapes 内容
    for line in txtList:  # 正样本(txt内容不为空)
        oneline = line.strip().split(" ")  # oneline是一个list, e.g. ['0', '0.31188484251968507', 
                                           #                         '0.6746135899679205', 
                                           #                         '0.028297244094488208', 
                                           #                         '0.04738990959463407']
        # 获取坐标并转换为左上和右下的形式
        x_center, y_center, w, h = float(oneline[1]), float(oneline[2]), float(oneline[3]), float(oneline[4])
        
        xmin = x_center - w/2
        ymin = y_center - h/2
        xmax = x_center + w/2
        ymax = y_center + h/2
        
        # 添加到 shapes 列表中
        json_data["shapes"].append({
            "label": oneline[0],
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

if SUCCEED_NUM + SKIP_NUM == TOTAL_NUM:
    print(f"\n👌 \033[1;32mNo Problem\033[0m")
else:
    print(f"\n🤡 \033[1;31m貌似有点问题, 请仔细核查!\033[0m")
