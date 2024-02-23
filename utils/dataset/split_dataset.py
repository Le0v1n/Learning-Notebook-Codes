import os
import shutil
import tqdm
from sklearn.model_selection import train_test_split
import logging
import datetime
import sys

sys.path.append(os.getcwd())
from utils.common_fn import print_arguments, xprint


__doc__ = """将数据集按比例进行随机划分
    数据集结构说明：
        example_dataset  # 数据集名称
        └── raw_data     # 未划分的数据
            ├── images   # 未划分的图片
            └── label    # 未划分的标签
    生成的数据集结构
        example_dataset  # 数据集名称
        ├── raw_data     # 未划分的数据
        │   ├── images   # 未划分的图片
        │   └── labels   # 未划分的标签
        ├── train        # 划分好的训练集
        │   ├── images
        │   └── labels
        └── val          # 划分好的验证集
            ├── images
            └── labels
"""
xprint(__doc__, color='blue', bold=True, horizontal_line="=", horizontal_line_num=2)


"""============================ 需要修改的地方 ==================================="""
BASE_PATH = 'Datasets/coco128'
val_size = 0.2  # 验证集大小(可以是数字也可以是浮点数)

IMAGE_TYPE = ('.jpg', '.png')  # 图片数据类型
LABEL_TYPE = ('.txt', )  # 标签的数据类型

random_seed = 42  # 随机数种子
LOG_FOLDER_NAME = "local-log"  # 存放日志的文件夹名称
"""==============================================================================="""

# 组合路径
images_path = os.path.join(BASE_PATH, 'train', 'images')  # 未划分图片路径
labels_path = os.path.join(BASE_PATH, 'train', 'labels')  # 未划分标签路径

train_images_save_path = os.path.join(BASE_PATH, "train", "images")  # 划分好的训练集图片保存路径
train_labels_save_path = os.path.join(BASE_PATH, "train", "labels")  # 划分好的训练集标签保存路径
val_images_save_path = os.path.join(BASE_PATH, "val", "images")  # 划分好的验证集图片保存路径
val_labels_save_path = os.path.join(BASE_PATH, "val", "labels")  # 划分好的验证集标签保存路径

"---------------------------------------日志---------------------------------------"
script_path = os.path.abspath(__file__)  # 获取当前脚本的绝对路径
script_name = os.path.splitext(os.path.basename(script_path))[0]  # 当前脚本的名称(没有.py后缀)
script_folder_path = os.path.dirname(script_path)  # 获取当前脚本所在的文件夹名
log_folder_path = os.path.join(script_folder_path, LOG_FOLDER_NAME)  # 存放log的文件夹路径

formatted_time = datetime.datetime.now().strftime("%Y%m%d-%H_%M_%S")  # 获取当前时间并格式化为指定格式
log_filename = os.path.join(log_folder_path, formatted_time + '-' + script_name + '.log')   # 获取文件夹名并拼接日志文件名
log_file_path = os.path.join(script_folder_path, log_filename)  # 拼接日志文件的完整路径
"---------------------------------------------------------------------------------"

# 读取所有数据(图片 + 标签)
images = [os.path.join(images_path, file) for file in os.listdir(images_path) if file.lower().endswith(IMAGE_TYPE)]
labels = [os.path.join(labels_path, file) for file in os.listdir(labels_path) if file.lower().endswith(LABEL_TYPE)]

"------------计数------------"
images_num = len(images)
labels_num = len(labels)
"---------------------------"

assert images_num == labels_num, f"图片和标签数量不一致!({images_num} != {labels_num})"

# 计算训练集和验证集大小
_test_size_really = 0
if 0 < val_size < 1:  # 浮点数
    test_samples_num = int(images_num * val_size)
    train_samples_num = images_num - test_samples_num
    _test_size_really = val_size
elif val_size > 1:  # 整数
    test_samples_num = int(val_size)
    train_samples_num = images_num - test_samples_num
    _test_size_really = val_size / images_num
elif val_size == 0:
    test_samples_num = 0
    train_samples_num = images_num - test_samples_num
    _test_size_really = 0
elif val_size < 0:
    raise ValueError("验证集数量不能为负!")

xprint("⚠️  验证集数量为 0, 不推荐!", color='red', bold=True) if test_samples_num == 0 else ...
param_dict = dict(
    图片路径=images_path,
    标签路径=labels_path,
    图片数量=images_num,
    标签数量=labels_num,
    训练集大小=train_samples_num,
    验证集大小=test_samples_num,
    训练集图片保存路径=train_images_save_path,
    训练集标签保存路径=train_labels_save_path,
    验证集图片保存路径=val_images_save_path,
    验证集标签保存路径=val_labels_save_path,
    日志保存路径=log_file_path,
    wait=True
)
table = print_arguments(**param_dict) 

# 配置日志输出的格式和级别
os.mkdir(log_folder_path) if not os.path.exists(log_folder_path) else ...
logging.basicConfig(filename=log_file_path, 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 创建日志记录器
logger = logging.getLogger()
logger.info(f"\n{table}")

# 创建控制台处理器并添加到日志记录器
# console_handler = logging.StreamHandler()
# logger.addHandler(console_handler)

# 使用sklearn进行数据集划分
if _test_size_really != 0:
    train_images, val_images = train_test_split(images, 
                                                test_size=_test_size_really, 
                                                random_state=random_seed)
    print(f"训练集实际大小: {len(train_images)}/{images_num}"
          f"| 验证集实际大小: {len(val_images)}/{images_num}")
else:
    train_images = images
    val_images = []

# 生成对应的标签列表
train_labels = []
val_labels = []

_no_labeled_images = []  # 存放没有标签的图片
for tip in train_images:  # tip -> train_img_path
    pre, _ = os.path.splitext(os.path.basename(tip))
    _path = os.path.join(train_labels_save_path, pre + ".txt")
    
    # 如果不是训练集，那么就应该在验证集
    if not os.path.exists(_path):  
        _path = os.path.join(val_labels_save_path, pre + ".txt")
    
    # 再次判断是否存在
    if not os.path.exists(_path):
        _no_labeled_images.append(tip)
        logger.error(f"⚠️ 标签不存在: {tip}")
        
    # 没啥问题
    train_labels.append(_path)

for vip in val_images:  # vip -> val_img_path
    pre, _ = os.path.splitext(os.path.basename(vip))
    _path = os.path.join(train_labels_save_path, pre + ".txt")
    
    # 如果不是训练集，那么就应该在验证集
    if not os.path.exists(_path):  
        _path = os.path.join(val_labels_save_path, pre + ".txt")
    
    # 再次判断是否存在
    if not os.path.exists(_path):
        _no_labeled_images.append(vip)
        logger.error(f"⚠️ 标签不存在: {vip}")
        
    # 没啥问题
    val_labels.append(_path)
        
if _no_labeled_images:
    logger.error("\n部分数据标签不存在, 请处理后再操作!")
for ip in _no_labeled_images:  # ip -> image_path
    cp_folder_path = os.path.join(os.path.dirname((os.path.dirname(os.path.dirname(ip)))), "no-labeled-images")
    os.mkdir(cp_folder_path) if not os.path.exists(cp_folder_path) else ...
    shutil.copy(src=ip, dst=cp_folder_path)
if _no_labeled_images:
    xprint(f"已复制不存在标签的图片，请进行标注！\n"
           f"路径为: {cp_folder_path}", color='red', bold=True)
    exit()

assert len(train_images) == len(train_labels), f"训练集图片数量和标签数量不一致!({len(train_images)}/{len(train_labels)})"
assert len(val_images) == len(val_labels), f"训练集图片数量和标签数量不一致!({len(val_images)}/{len(val_labels)})"

# 定义保存训练集和验证集的文件夹路径
train_image_save_folder = train_images_save_path if len(train_images) != 0 else ...
train_label_save_folder = train_labels_save_path if len(train_labels) != 0 else ...
val_image_save_folder = val_images_save_path if len(val_images) != 0 else ...
val_label_save_folder = val_labels_save_path if len(val_labels) != 0 else ...

# 创建保存文件夹
os.mkdir(train_image_save_folder) if not os.path.exists(train_image_save_folder) else ...
os.mkdir(train_label_save_folder) if not os.path.exists(train_label_save_folder) else ...
os.makedirs(val_image_save_folder) if not os.path.exists(val_image_save_folder) else ...
os.makedirs(val_label_save_folder) if not os.path.exists(val_label_save_folder) else ...

# 将图片和标签拷贝到对应文件夹
progress_bar = tqdm.tqdm(total=len(train_images) + len(val_images), desc="训练集 + 验证集", unit=" file")
for (ip, lp) in zip(train_images, train_labels):  # ip --> image_path; lp --> label_path
    image_base_name = os.path.basename(ip)
    label_base_name = os.path.basename(lp)
    image_dst_path = os.path.join(train_image_save_folder, image_base_name)
    label_dst_path = os.path.join(train_label_save_folder, label_base_name)

    # 开始移动
    shutil.move(src=ip, dst=image_dst_path) if ip != image_dst_path else ...
    shutil.move(src=lp, dst=label_dst_path) if lp != label_dst_path else ...
    progress_bar.update(1)
    
for (ip, lp) in zip(val_images, val_labels):  # ip --> image_path; lp --> label_path
    image_base_name = os.path.basename(ip)
    label_base_name = os.path.basename(lp)
    image_dst_path = os.path.join(val_image_save_folder, image_base_name)
    label_dst_path = os.path.join(val_label_save_folder, label_base_name)

    # 开始移动
    shutil.move(src=ip, dst=image_dst_path) if ip != image_dst_path else ...
    shutil.move(src=lp, dst=label_dst_path) if lp != label_dst_path else ...
    progress_bar.update(1)
progress_bar.close()

# ================================= 重新遍历一遍 =================================
train_images = [file for file in os.listdir(train_image_save_folder) if file.lower().endswith(IMAGE_TYPE)]
train_labels = [file for file in os.listdir(train_label_save_folder) if file.lower().endswith(LABEL_TYPE)]
val_images = [file for file in os.listdir(val_image_save_folder) if file.lower().endswith(IMAGE_TYPE)]
val_labels = [file for file in os.listdir(val_label_save_folder) if file.lower().endswith(LABEL_TYPE)]
# ================================================================================

result_dict = dict(
    训练集图片保存路径=train_image_save_folder,
    训练集标签保存路径=train_label_save_folder,
    验证集图片保存路径=val_image_save_folder,
    验证集标签保存路径=val_label_save_folder,
    训练集图片数量=len(train_images),
    训练集标签数量=len(train_labels),
    验证集图片数量=len(val_images),
    验证集标签数量=len(val_labels),
    日志保存路径=log_file_path,
)

table = print_arguments(**result_dict)

logger.info(f"\n{table}")

if (len(train_images) + len(val_images) == images_num) and (len(train_labels) + len(val_labels) == labels_num):
    _str = (f"👌 No Problems in data numbers")
    logger.info(_str)
    xprint(_str, color='green')

_str = "Finished!"
logger.info(_str)
xprint(_str, color='green', underline=True, horizontal_line='>', bold=True)