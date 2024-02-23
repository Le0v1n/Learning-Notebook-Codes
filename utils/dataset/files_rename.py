import os
import sys
import tqdm
import datetime

sys.path.append(os.getcwd())
from utils.common_fn import print_arguments, xprint
from utils.file_type import ImageFormat, LabelFormat, VideoFormat


__doc__ = """脚本说明：可以自动重命名某个文件夹下指定类型的文件
    修改前文件名称: img1.jpg
    修改后文件名称: Le0v1n-20231123-X-0001.jpg
    用途：统一文件名称
    要求：无
"""
xprint(__doc__, color='blue', bold=True, horizontal_line="=", horizontal_line_num=2)


"""============================ 需要修改的地方 ==================================="""
src_path = 'utils/dataset/EXAMPLE_FOLDER'  # 目标文件夹路径
file_type = ImageFormat  # 重命名文件的文件类型

# -------------------重命名相关------------------
retain_previous_name = False  # 是否保留之前的名称
new_name = "Le0v1n"  # retain_previous_name为False时生效
use_date_stamp = True  # 是否使用时间戳 -> e.g. 20231123
comment = "X"  # 备注
use_serial_numbering = True  # 是否使用顺序的编号 -> 1, 2, 3, 4, 5, 6, ...
start_number = 1  # 从编号几开始 -> e.g. 1: 从 0001 开始编号
numbering_placeholder = 4  # 编号保留的占位 -> e.g. 0001, 0002, 0003, ...
hyphen = '-'  # 连字符 -> e.g. filename-0001.jpg
"""==============================================================================="""

print_arguments(
    是否保留之前的名称=retain_previous_name,
    新名称=new_name,
    是否使用时间戳=use_date_stamp,
    备注=comment,
    是否使用顺序的编号=use_serial_numbering,
    编号起始值=start_number,
    编号保留的占位=numbering_placeholder,
    连字符=hyphen,
    confirm=True
)

# 获取目录中的所有图片文件
files_list = [file for file in os.listdir(src_path) if file.lower().endswith(file_type)]

"------------计数------------"
TOTAL_FILES_NUM = len(files_list)  # 需要重命名的文件数量
RENAME_NUM = 0  # 重命名成功数量
"---------------------------"

# 获取当前时间并格式化时间戳
timestamp = datetime.datetime.now().strftime("%Y%m%d")

# 遍历文件
process_bar = tqdm.tqdm(total=TOTAL_FILES_NUM, desc="为指定格式的文件重命名", unit='file')  # 创建进度条
for idx, file_name in enumerate(files_list):
    file_pre, file_ext = os.path.splitext(file_name)  # 获得文件名和后缀
    process_bar.set_description(f"rename for \033[1;31m{file_name}\033[0m")

    # 构建新的文件名
    if retain_previous_name:  # 保留原有的名称
        NEW_FILE_NAME = f"{file_pre}"
    elif new_name:  # 不保留原有的名称且新名称存在
        NEW_FILE_NAME = new_name
    else:  # 不保留原有的名称也没有新名称 -> 报错
        raise KeyError(f"不保留原有的名称也没有新名称!")
    
    if use_date_stamp:  # 使用时间戳
        NEW_FILE_NAME += f"{hyphen}{timestamp}"
    
    if comment:  # 添加备注
        NEW_FILE_NAME += f"{hyphen}{comment}" 
    
    if use_serial_numbering:  # 使用编号
        NEW_FILE_NAME += f"{hyphen}{idx + start_number:0{numbering_placeholder}d}"

    # 加上扩展名
    NEW_FILE_NAME += file_ext
    
    # 开始重命名文件         
    _src = os.path.join(src_path, file_name)  # 旧文件路径
    _dst = os.path.join(src_path, NEW_FILE_NAME)  # 新文件路径
    
    os.rename(_src, _dst)  # 重命名文件
    RENAME_NUM += 1
    process_bar.update(1)
process_bar.close()
    
xprint(f"✔️  文件重命名完成: {RENAME_NUM}/{TOTAL_FILES_NUM}"
       f"--> {src_path}", color='green', bold=True, horizontal_line='<')
