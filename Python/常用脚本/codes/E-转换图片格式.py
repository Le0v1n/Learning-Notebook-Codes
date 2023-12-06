"""
+ 脚本说明：对指定文件夹下所有的图片进行格式转换
+ 用途：统一数据集图片的格式
+ 要求：无
+ 注意：
  1. 不需要转换的则跳过
  2. 不是图片的文件有两种操作方式：
    2.1 mv/move 扔到 RECYCLE_BIN_PATH
    2.2 del/delete 直接删除
"""
import os
import tqdm
import shutil
from PIL import Image


"""============================ 需要修改的地方 ==================================="""
IMAGE_PATH = "EXAMPLE_FOLDER/images"  # 输入图片所在文件夹路径
IMAGE_TYPE = ('.jpg', '.png', '.jpeg')  # 哪些类型的文件会被转换
CONVERT_TYPE = '.jpg'  # 想要转换为什么格式: .jpg .png
OPERATION_METHOD = 'mv'  # 操作的方式: mv or del
"""==============================================================================="""

if CONVERT_TYPE == '.jpg':
    _convert_type_in_pil = 'JPEG'
elif CONVERT_TYPE == '.png':
    _convert_type_in_pil = 'PNG'
elif CONVERT_TYPE == '.gif':
    _convert_type_in_pil = 'GIF'
else:
    raise KeyError("只能转换为 [.jpg, .png, .gif]")

# 获取所有图片和标签
image_list = os.listdir(IMAGE_PATH)

# 过滤只包括特定类型的图像文件（这里是.jpg和.png）
image_list = [file for file in image_list if file.lower().endswith(IMAGE_TYPE)]  # 需要转换的图片list
DONT_CONVERT_LIST = [file for file in image_list if file.lower().endswith(CONVERT_TYPE)]  # 不需要转换的图片list

"------------计数------------"
TOTAL_NUM = len(image_list)
DONT_CONVERT_NUM = len(DONT_CONVERT_LIST)
SUCCEED_NUM = 0
SKIP_NUM = 0
"---------------------------"

del DONT_CONVERT_LIST  # 删除list释放空间

if DONT_CONVERT_NUM == TOTAL_NUM:  # 如果没有要转换的图片
    print(f"所有图片均为 {CONVERT_TYPE}({DONT_CONVERT_NUM}张), 无需转换, 程序停止!")
    exit()

_str = (f"💡 图片路径: \033[1;33m{IMAGE_PATH}\033[0m"
        f"\n💡 转换格式为: \033[1;33m{CONVERT_TYPE}\033[0m"
        f"\n💡 操作方式为: \033[1;33m{OPERATION_METHOD}\033[0m"
        f"\n 所有图片数量: \033[1;33m{TOTAL_NUM}\033[0m"
        f"\n 已有{CONVERT_TYPE} 图片数量: \033[1;33m{DONT_CONVERT_NUM}\033[0m"
        f"\n💡 预计转换图片数量: \033[1;33m{TOTAL_NUM - DONT_CONVERT_NUM}\033[0m"
        f"\n\n请输入 \033[1;31m'yes'\033[0m 继续，输入其他停止")
print(_str)

_INPUT = input()
if _INPUT != "yes":
    exit()
    
if OPERATION_METHOD in ('mv', 'move'):
    RECYCLE_BIN_PATH = os.path.join(os.path.dirname(IMAGE_PATH), "Recycle_bins")  # 将垃圾桶放在图片的上一级目录中
    if not os.path.exists(RECYCLE_BIN_PATH):
        os.mkdir(RECYCLE_BIN_PATH)

# 创建进度条
progress_bar = tqdm.tqdm(total=TOTAL_NUM, desc=f"Convert {IMAGE_TYPE} into {CONVERT_TYPE}", unit="img")
for image_name in image_list:
    pre, ext = os.path.splitext(image_name)  # 分离文件名和后缀

    # 如果是我们想要的图片格式则跳过
    if ext == CONVERT_TYPE:
        SKIP_NUM += 1
        progress_bar.update()
        continue
    
    # 需要转换
    image_path = os.path.join(IMAGE_PATH, image_name)  # 被转换图片的完整路径
    image_save_path = os.path.join(IMAGE_PATH, pre) + CONVERT_TYPE  # 转换后图片的保存路径
    
    # 打开图片并进行转换
    image = Image.open(image_path)
    image.save(image_save_path, _convert_type_in_pil)
    
    # 开始操作原有的图片: 移到垃圾桶还是直接删除
    if OPERATION_METHOD in ('mv', 'move'):
        dst_path = os.path.join(RECYCLE_BIN_PATH, image_name)
        shutil.move(src=image_path, dst=dst_path)
    elif OPERATION_METHOD in ('del', 'delete'):
        os.remove(image_path)
        
    SUCCEED_NUM += 1
    progress_bar.update(1)
progress_bar.close()

_str = (f"👌 将图片转换为{CONVERT_TYPE}已完成，详情如下："
        f"\n\t成功转换的图片数量: \033[1;32m{SUCCEED_NUM}/{TOTAL_NUM}\033[0m"
        f"\n\t跳过转换的图片数量: \033[1;33m{SKIP_NUM}/{TOTAL_NUM}\033[0m"
        f"\n\t转换后的图片路径为: \033[1;34m{IMAGE_PATH}\033[0m")
_str += f"\n\t垃圾桶路径为: \033[1;34m{RECYCLE_BIN_PATH}\033[0m" if OPERATION_METHOD in ('mv', 'move') else _str
print(_str)

if SUCCEED_NUM + SKIP_NUM == TOTAL_NUM and SKIP_NUM == DONT_CONVERT_NUM:
    print(f"👌 No problems")
