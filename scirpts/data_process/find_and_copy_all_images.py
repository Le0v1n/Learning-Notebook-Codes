"""
递归遍历一个文件夹，将其所有的图片都保存到指定的文件夹中
"""
import os
import shutil
from pathlib import Path

def copy_images(source_path, target_path):
    # 确保目标路径存在
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    # 支持的图片格式
    image_extensions = ('.jpg', '.jpeg', '.png')
    
    # 遍历源路径下的所有文件和文件夹
    for root, dirs, files in os.walk(source_path):
        for file in files:
            # 检查文件扩展名是否为图片格式
            if file.lower().endswith(image_extensions):
                # 构建完整的文件路径
                file_path = os.path.join(root, file)
                # 构建目标文件路径
                target_file_path = os.path.join(target_path, file)
                # 复制文件
                shutil.copy(file_path, target_file_path)
                print(f"Copied: {file_path} to {target_file_path}")

# 输入源文件夹路径
source_folder = "/mnt/d/znv/项目/数据处理/杭州联通数据/杭州联通数据-minIO"
# 输入目标文件夹路径
target_folder = "/mnt/d/znv/项目/数据处理/杭州联通数据/杭州联通数据-minIO/所有的图片"
Path(target_folder).mkdir(exist_ok=True)

# 执行复制操作
copy_images(source_folder, target_folder)