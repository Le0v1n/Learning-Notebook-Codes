"""
    生成数据集
"""
# 导入所需库
import os
from sklearn.model_selection import train_test_split
import shutil
import tqdm


"""============================ 需要修改的地方 ==================================="""
test_size = 0.01
OVERRIDE = False

# 图片文件夹路径
target_image_folder = "EXAMPLE_DATASET/VOC2007/JPEGImages"

# txt文件夹路径
target_label_folder = "EXAMPLE_DATASET/VOC2007/labels"

# 输入文件夹路径
output_folder = "EXAMPLE_DATASET"
"""==============================================================================="""

# 读取所有.txt文件
labels = [label for label in os.listdir(target_label_folder) if label.endswith(".txt")]

TOTAL_NUM = len(labels)

print(f"预计验证集样本数量为: \033[1;31m{round(TOTAL_NUM * test_size)}\033[0m，请输入 \033[1;31myes\033[0m 继续 | 输入其他退出")

_INPUT = input()
if _INPUT != "yes":
    exit()

# 使用sklearn进行数据集划分
train_list, val_list = train_test_split(labels, test_size=test_size, random_state=42)
print(f"训练集大小: {len(train_list)}/{TOTAL_NUM} | 验证集大小: {len(val_list)}/{TOTAL_NUM}")

# 定义保存训练集和验证集的文件夹路径
train_image_folder = os.path.join(output_folder, "train", "images")
train_label_folder = os.path.join(output_folder, "train", "labels")
val_image_folder = os.path.join(output_folder, "val", "images")
val_label_folder = os.path.join(output_folder, "val", "labels")
print(f"train_image_folder: {train_image_folder}")
print(f"train_label_folder: {train_label_folder}")
print(f"val_image_folder: {val_image_folder}")
print(f"val_label_folder: {val_label_folder}")

# 创建保存文件夹
os.makedirs(train_image_folder, exist_ok=True)
os.makedirs(train_label_folder, exist_ok=True)
os.makedirs(val_image_folder, exist_ok=True)
os.makedirs(val_label_folder, exist_ok=True)

print("=" * 50)

# 将训练集的图片和标签拷贝到对应文件夹
progress_bar = tqdm.tqdm(total=len(train_list), desc="Copying in \033[1;31mtrain\033[0m", unit=" file")
TRAIN_SUCCESSES_NUM = 0
TRAIN_SKIP_NUM = 0
for label in train_list:
    label_path = os.path.join(target_label_folder, label)
    image_path = os.path.join(target_image_folder, label.replace(".txt", ".jpg"))
    
    # 定义目标路径
    target_img = os.path.join(train_image_folder, label.replace(".txt", ".jpg"))
    target_label = os.path.join(train_label_folder, label)
    if not OVERRIDE and os.path.exists(target_img) and target_label:
        TRAIN_SKIP_NUM += 1
        progress_bar.update(1)
        continue

    shutil.copy(image_path, target_img)
    shutil.copy(label_path, target_label)
    TRAIN_SUCCESSES_NUM += 1
    progress_bar.update(1)
progress_bar.close()

# 将验证集的图片和标签拷贝到对应文件夹
progress_bar = tqdm.tqdm(total=len(train_list), desc="Copying in \033[1;31mvalidation\033[0m", unit=" file")
VAL_SUCCESSES_NUM = 0
VAL_SKIP_NUM = 0
for label in val_list:
    label_path = os.path.join(target_label_folder, label)
    image_path = os.path.join(target_image_folder, label.replace(".txt", ".jpg"))

    # 定义目标路径
    target_img = os.path.join(val_image_folder, label.replace(".txt", ".jpg"))
    target_label = os.path.join(val_label_folder, label)
    
    if not OVERRIDE and os.path.exists(target_img) and target_label:
        VAL_SKIP_NUM += 1
        progress_bar.update(1)
        continue

    shutil.copy(image_path, target_img)
    shutil.copy(label_path, target_label)
    VAL_SUCCESSES_NUM += 1
    progress_bar.update(1)
progress_bar.close()

print(
    f"\n数据集创建完毕，详情如下：\n\t"
    f"训练集:\n\t\t"
    f"图片路径: {train_image_folder}\n\t\t"
    f"标签路径: {train_label_folder}\n\t\t\t"
    f"👌成功: {TRAIN_SUCCESSES_NUM}/{len(train_list)}\n\t\t\t"
    f"👌跳过: {TRAIN_SKIP_NUM}/{len(train_list)}\n\t"
    
    f"验证集:\n\t\t"
    f"图片路径: {val_image_folder}\n\t\t"
    f"标签路径: {val_label_folder}\n\t\t\t"
    f"👌成功: {VAL_SUCCESSES_NUM}/{len(val_list)}\n\t\t\t"
    f"👌跳过: {VAL_SKIP_NUM}/{len(val_list)}"
)