"""
    json转yolo的txt
"""

import os
import cv2
import json
import numpy as np
import tqdm

"""============================ 需要修改的地方 ==================================="""
# 标签字典
label_dict = {'cls_1': 0,
              'cls_2': 1,
              }
# 文件夹路径
base_path = 'EXAMPLE_DATASET/VOC2007'

OVERRIDE = False  # 是否要覆盖已存在txt文件
use_kpt_check = False
"""==============================================================================="""

path = os.path.join(base_path, 'Annotations')
all_json_list = os.listdir(path)
TOTAL_NUM = len(all_json_list)
SUCCESSES_NUM = 0
SKIP_NUM = 0
ERROR_NUM = 0
ERROR_LIST = []

# 创建进度条
progress_bar = tqdm.tqdm(total=len(all_json_list), desc="json2yolo", unit=" .txt")

for idx, anno_name in enumerate(all_json_list):  # anno_json = 'xxxxxx_yyyyyyy_ccccc.json'
    target_path = os.path.join(base_path, 'labels', anno_name.replace('.json', '.txt'))
    if not OVERRIDE and os.path.exists(target_path):
        SKIP_NUM += 1
        continue

    progress_bar.set_description(f"\033[1;31m{anno_name}\033[0m")

    with open(os.path.join(path, anno_name), 'r') as fr:
        result = json.load(fr)

    img = cv2.imread(os.path.join(base_path, 'JPEGImages',
                     anno_name).replace('.json', '.jpg'))
    h_, w_ = img.shape[0:2]
    object_info = result['shapes']

    # exist_ok=True 表示如果目标目录已存在，则不会引发异常，而是默默地跳过创建该目录的步骤
    os.makedirs(os.path.join(base_path, 'labels'), exist_ok=True)
    with open(target_path, 'w') as target_file:
        try:
            for line in object_info:
                label = label_dict[line['label']]
                # label = 0 if line['label'] == 'chepai' else 1
                kpt = np.array(line['points'])
                if use_kpt_check and (kpt[1][0] > kpt[3][0] and kpt[1][1] > kpt[3][1]):
                    continue
                else:
                    x1, y1, x2, y2 = kpt[0][0], kpt[0][1], kpt[1][0], kpt[1][1]
                    xc, yc, w, h = x1 + (x2-x1)/2, y1 + (y2-y1)/2, x2-x1, y2-y1

                    line = '{} {} {} {} {}'.format(
                        label, xc/w_, yc/h_, w/w_, h/h_)
                    target_file.write(line+'\n')
            SUCCESSES_NUM += 1

        except:
            ERROR_NUM += 1
            ERROR_LIST.append(os.path.join(path, anno_name))

    progress_bar.update(1)
progress_bar.close()

for _ef in ERROR_LIST:
    print(_ef)

print(f"json2yolo已完成，详情如下：\n\t"
      f"👌成功: {SUCCESSES_NUM}/{TOTAL_NUM}\n\t"
      f"👌跳过: {SKIP_NUM}/{TOTAL_NUM}\n\t"
      f"🤡失败: {ERROR_NUM}/{TOTAL_NUM}")