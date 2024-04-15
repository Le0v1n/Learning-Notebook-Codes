"""
    描述：为所有图片创建空的json文件（如果json文件存在则跳过）
    作用：为负样本生成对应的json文件
"""

import numpy as np
import os
import cv2
import json
import tqdm


"""============================ 需要修改的地方 ==================================="""
# 图片所在文件夹路径
source_folder_path = 'EXAMPLE_DATASET/VOC2007/JPEGImages'

# json文件路径
target_folder_path = 'EXAMPLE_DATASET/VOC2007/Annotations'

# 负样本数量
NEG_SAMPLE_NUM = 1024
"""==============================================================================="""

# 获取所有图片
image_list = os.listdir(source_folder_path)
# 过滤只包括特定类型的图像文件（这里是.jpg和.png）
image_list = [file for file in image_list if file.lower().endswith(('.jpg', '.png'))]
TOTAL_NUM = len(image_list)
SKIP_NUM = 0
SUCCEED_NUM = 0

# 创建进度条
progress_bar = tqdm.tqdm(total=len(image_list), desc="json2yolo", unit=" .json")
for image_name in image_list:
    # 分离文件名和后缀
    image_pre, image_ext = os.path.splitext(image_name)

    # 确定保存的路径
    target_path = os.path.join(target_folder_path, image_pre) + '.json'
    # 确定图片的路径
    img_file = os.path.join(source_folder_path, image_name)

    # 判断对应的json文件是否存在
    if os.path.exists(target_path):
        SKIP_NUM += 1
        progress_bar.update(1)
        continue

    img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_COLOR)
    height, width, _ = img.shape
    content = {"version": "0.2.2",
               "flags": {},
               "shapes": [],
               "imagePath": "{}.jpg".format(image_pre),
               "imageData": None,
               "imageHeight": height,
               "imageWidth": width
               }
    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)

    with open(target_path, 'w') as f:
        json.dump(content, f, indent=2)
    SUCCEED_NUM += 1
    progress_bar.update(1)

print(f"SUCCEED NUM: {SUCCEED_NUM}/{TOTAL_NUM}")
print(f"SKIP NUM: {SKIP_NUM}/{TOTAL_NUM}")

if SUCCEED_NUM == NEG_SAMPLE_NUM:
    print("\n\033[1;36m👌No Problems\033[0m\n")
else:
    print(f"\n\033[1;31m🤡有问题: 成功/负样本数量 -> {SUCCEED_NUM}/{NEG_SAMPLE_NUM}\033[0m\n")