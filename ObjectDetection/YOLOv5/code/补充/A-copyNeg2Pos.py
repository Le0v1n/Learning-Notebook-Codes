"""
    描述：
        1. 检查负样本数量是否正确；
        2. 检查正样本数量是否正确；
        3. 检查Annotations数量是否正确
"""
import os
import shutil
import tqdm


"""============================ 需要修改的地方 ==================================="""
# 数据所在路径
BASE_PATH = 'EXAMPLE_DATASET/DATASET_A'
CHECK_NUM = False  # 是否检查样本数量
POS_SAMPLE_NUM = 6914  # 正样本数量 -> 6914
NEG_SAMPLE_NUM = 515  # 负样本数量 -> 515
"""==============================================================================="""

# 组合路径
source_path = os.path.join(BASE_PATH, "VOC2007")  # EXAMPLE_DATASET/VOC2007
pos_image_path = os.path.join(source_path, "JPEGImages")  # EXAMPLE_DATASET/VOC2007/JPEGImages
annotation_path = os.path.join(source_path, "Annotations")  # EXAMPLE_DATASET/VOC2007/Annotations
neg_image_path = os.path.join(source_path, "neg_samples")  # EXAMPLE_DATASET/VOC2007/neg_samples

# 获取所有图片和标签
pos_image_list = os.listdir(pos_image_path)
annotation_list = os.listdir(annotation_path)
neg_image_list = os.listdir(neg_image_path)

# 过滤只包括特定类型的图像文件（这里是.jpg和.png）
pos_image_list = [file for file in pos_image_list if file.lower().endswith(('.jpg', '.png'))]
annotation_list = [file for file in annotation_list if file.lower().endswith(('.json', '.xml'))]
neg_image_list = [file for file in neg_image_list if file.lower().endswith(('.jpg', '.png'))]

# 记录实际数据数量
POS_IMG_NUM = len(pos_image_list)
ANNOTATIONS_NUM = len(annotation_list)
NEG_IMG_NUM = len(neg_image_list)

# 检查数据是否正确
if CHECK_NUM:
    assert POS_SAMPLE_NUM == POS_IMG_NUM, f"\033[1;31m正样本数量({POS_SAMPLE_NUM})和实际正样本数量({POS_IMG_NUM})不一致！\033[0m"
    assert CHECK_NUM and POS_IMG_NUM == ANNOTATIONS_NUM, f"\033[1;31m实际正样本数量({POS_IMG_NUM})和实际标签数量({ANNOTATIONS_NUM})不一致！\033[0m"
    assert CHECK_NUM and NEG_SAMPLE_NUM == NEG_IMG_NUM, f"\033[1;31m负样本数量({NEG_SAMPLE_NUM})和实际负样本数量({NEG_IMG_NUM})不一致！\033[0m"
else:
    print("\033[1;31m💡请注意：跳过了数据检查！\033[0m")

SKIP_NUM = 0
SUCCEED_NUM = 0

# 创建进度条
progress_bar = tqdm.tqdm(total=NEG_IMG_NUM, desc="Copy neg2pos", unit=" img")
for neg_image_name in neg_image_list:
    # 分离文件名和后缀
    image_pre, image_ext = os.path.splitext(neg_image_name)

    # 确定图片的路径 -> EXAMPLE_DATASET/VOC2007/neg_samples/xxxx_yyyy_xxxx_yyyy.jpg
    src_img_path = os.path.join(neg_image_path, neg_image_name)
    # 确定保存的路径 -> EXAMPLE_DATASET/VOC2007/JPEGImages/xxxx_yyyy_xxxx_yyyy.jpg
    target_img_path = os.path.join(pos_image_path, neg_image_name)

    # 判断对应的json文件是否存在
    if os.path.exists(target_img_path):
        SKIP_NUM += 1
        progress_bar.update(1)
        continue
    
    # 开始复制
    shutil.copy(src=src_img_path, dst=target_img_path)
    SUCCEED_NUM += 1
    progress_bar.update(1)

print(f"SUCCEED NUM: {SUCCEED_NUM}/{NEG_IMG_NUM}")
print(f"SKIP NUM: {SKIP_NUM}/{NEG_IMG_NUM}")

if SUCCEED_NUM + SKIP_NUM == NEG_SAMPLE_NUM:
    print("\n\033[1;36mNo Problems in Copying\033[0m\n")
    # 再次检查数据数量
    if POS_SAMPLE_NUM + NEG_SAMPLE_NUM == POS_IMG_NUM + SUCCEED_NUM:
        print(f"\n\033[1;36m👌预想正负样本数量({POS_SAMPLE_NUM} + {NEG_SAMPLE_NUM}) == 实际的正负样本数量({POS_IMG_NUM} + {SUCCEED_NUM})\033[0m\n")
    else:
        print(f"\n\033[1;31m🤡出现了问题：预想正负样本数量({POS_SAMPLE_NUM} + {NEG_SAMPLE_NUM}) != 实际的正负样本数量({POS_IMG_NUM} + {SUCCEED_NUM})\033[0m\n")
else:
    print(f"\n\033[1;31m🤡有问题: 成功/负样本数量 -> {SUCCEED_NUM}/{NEG_SAMPLE_NUM}\033[0m\n")