import os
import sys
from PIL import Image

sys.path.append(os.getcwd())
from utils.common_fn import xprint, print_arguments


__doc__ = """压缩图片
    1. 读取一个文件夹中所有的图片（格式可能为.jpg、.png、webp）
    2. 判断图片大小，如果大于500kb，则将其大小缩小为500kb; 如果小于500kb则大小不用变
    3. 所有图片都保存为.jpg格式
    4. 删除对应的非.jpg格式的图片
"""


def resize_images(image_folder_path, target_size_kb=500):
    # 获取输入文件夹中的所有图片文件
    image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(('.jpg', '.png', '.webp'))]

    for image_name in image_files:
        name, ext = os.path.splitext(image_name)
        image_path = os.path.join(image_folder_path, image_name)
        save_path = os.path.join(image_folder_path, name) + '.jpg'

        # 打开图片
        with Image.open(image_path) as img:
            # 转换图片为RGB模式
            img = img.convert('RGB')
            
            # 获取图片大小（以KB为单位）
            size_kb = os.path.getsize(image_path) / 1024

            # 如果图片大小大于目标大小，则进行缩小
            if size_kb > target_size_kb:
                # 计算缩放比例
                scale_factor = (target_size_kb / size_kb) ** 0.5
                new_size = tuple(int(dim * scale_factor) for dim in img.size)

                # 缩小图片并保存为JPEG格式
                img.thumbnail(new_size, Image.LANCZOS)
                img.save(save_path, 'JPEG')
                xprint(f"Resized", color='red', end='')
                xprint(f": {image_name} - {size_kb}KB -> {os.path.getsize(save_path) / 1024}KB")
            else:
                # 复制原始图片到输出文件夹
                img.save(save_path, 'JPEG')
                xprint(f"Unchanged", color='green', end='')
                xprint(f": {image_name} - {size_kb}KB")
                
        if ext != '.jpg':  # 删除之前的非 .jpg 格式的图片
            os.remove(image_path)


if __name__ == "__main__":
    image_folder_path = "Datasets/Web/images"
    target_size_kb = 500
    
    print_arguments(
        图片路径=image_folder_path,
        目标大小_kb=target_size_kb,
        confirm=True
    )
    
    # 开始压缩
    resize_images(image_folder_path, target_size_kb)
    