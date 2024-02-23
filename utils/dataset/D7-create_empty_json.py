"""
+ 脚本说明：为负样本生成空的json文件，如果保存目录中有json文件则不创建（确保正样本的json文件不会被覆盖）
+ 用途：为负样本生成空的json文件
+ 要求：要有对应的图片文件，这样读取到的尺寸信息是最准确的。
+ 注意: 无。
"""
import numpy as np
import os
import cv2
import json
import tqdm


"""============================ 需要修改的地方 ==================================="""
IMAGE_PATH = 'EXAMPLE_FOLDER/images'  # 图片路径
JSON_PATH = 'EXAMPLE_FOLDER/annotations-json'  # json标签路径
IMAGE_TYPE = '.jpg'

# Json 文件基础信息
__version = "0.2.2"
__flags = {},
__shapes = [],
__imageData = None
"""==============================================================================="""

# 获取所有图片
image_list = [file for file in os.listdir(IMAGE_PATH) if file.endswith('.jpg')]
json_list = [file for file in os.listdir(JSON_PATH) if file.endswith('.json')]

"------------计数------------"
TOTAL_NUM = len(image_list)
SKIP_NUM = 0
SUCCEED_NUM = 0
"---------------------------"

print(f"💡 图片路径: \033[1;33m{IMAGE_PATH}\033[0m"
      f"\n💡 json路径: \033[1;33m{JSON_PATH}\033[0m"
      f"\n\t 图片数量: \033[1;33m{TOTAL_NUM}\033[0m"
      f"\n\t 目前json数量: \033[1;33m{len(json_list)}\033[0m"
      f"\n\t 预计生成的json数量: \033[1;33m{TOTAL_NUM - len(json_list)}\033[0m"
      f"\n\n请输入 \033[1;31m'yes'\033[0m 继续，输入其他停止"
      )
_INPUT = input()
if _INPUT != "yes":
    exit()

# 创建进度条
progress_bar = tqdm.tqdm(total=TOTAL_NUM, desc="为负样本创建json文件", unit=" .json")
for image_name in image_list:
    progress_bar.set_description(f"Process in \033[1;31m{image_name}\033[0m")
    # 分离文件名和后缀
    image_pre, image_ext = os.path.splitext(image_name)

    # 确定保存的路径
    json_path = os.path.join(JSON_PATH, image_pre) + '.json'
    # 确定图片的路径
    img_file = os.path.join(IMAGE_PATH, image_name)

    # 判断对应的json文件是否存在
    if os.path.exists(json_path):
        SKIP_NUM += 1
        progress_bar.update(1)
        continue

    # 读取图片获取尺寸信息
    img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_COLOR)
    height, width, _ = img.shape

    content = {"version": __version,
               "flags": __flags,
               "shapes": __shapes,
               "imagePath": f"{image_pre}.jpg",
               "imageData": __imageData,
               "imageHeight": height,
               "imageWidth": width
               }

    # 创建json文件并写入内容
    with open(json_path, 'w') as f:
        json.dump(content, f, indent=2)
    SUCCEED_NUM += 1
    progress_bar.update(1)
progress_bar.close()

print(f"为负样本创建json文件已完成，详情如下：\n\t"
      f"👌成功: {SUCCEED_NUM}/{TOTAL_NUM}\n\t"
      f"👌跳过: {SKIP_NUM}/{TOTAL_NUM}\n\t")

if SUCCEED_NUM + SKIP_NUM == TOTAL_NUM and SUCCEED_NUM == TOTAL_NUM - len(json_list):
    print(f"👌 No Problems")

