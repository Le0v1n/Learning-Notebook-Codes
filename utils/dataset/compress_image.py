import os
import sys
from tqdm.rich import tqdm
from PIL import Image, ImageOps, ImageFilter
import prettytable
from utils.generator import create_folder
from utils.getter import get_file_size

from utils.outer import print_arguments, xprint

sys.path.append(os.getcwd())
from utils.getter import get_current_time
from utils.items import ImageFormat


__doc__ = """压缩图片
    1. 读取一个文件夹中所有的图片，格式可能为 '.png', '.jpg', '.jpeg', '.gif', '.tiff', '.svg', 
                                           '.raw', 'webp', '.heic', '.heif'
    2. 压缩图片方式 keep_size=False
        keep_size=False：改变图片尺寸的进行压缩
        keep_size=True: 不改变图片的尺寸
    3. 判断图片大小，如果大于target_size_kb，则将其大小缩小为小于target_size_kb; 如果小于target_size_kb则大小不用变
    4. 图片保存路径 save_folder_name='compress_images': ./compress_images
    5. 所有图片保存格式save_format='retain'：
        save_format='retain'：图片的格式不变
        save_format='.xxx'：图片保存为.xxx格式
"""


def compress_images(image_folder_path, target_size_kb=512, keep_size=False, 
                  save_folder_path='compress_images', save_format='retain', 
                  log=True, verbose=True):
    """压缩图片文件大小
        keep_size=False：改变图片尺寸的进行压缩
        keep_size=True: 不改变图片的尺寸

    Args:
        image_folder_path (str): 图片所在文件夹路径
        target_size_kb (int, optional): 图片文件大小不超过多少kb. Defaults to 512.
        keep_size (bool, optional): 是否要保持图片的尺寸不变. Defaults to False.
        save_folder_path (str, optional): 图片保存的文件夹路径. Defaults to 'compress_images'.
        save_format (str, optional): 保存图片的格式，可以指定格式也可以使用'retain'保留图片原始格式. Defaults to 'retain'.
        log (bool, optional): 使用保存日志. Defaults to True.
        verbose (bool, optional): 是否确定参数是否正确. Defaults to True.
    """
    
    print_arguments(
        image_folder_path=image_folder_path,
        target_size_kb=target_size_kb,
        keep_size=keep_size,
        save_folder_name=save_folder_path,
        save_format=save_format,
        log=log,
        confirm=True
    ) if verbose else ...
    
    # 获取输入文件夹中的所有图片文件
    image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(ImageFormat)]
    create_folder(fp=save_folder_path)
    
    # 用于记录的信息
    log_image_paths = []
    log_image_save_paths = []
    log_file_sizes = []

    process_bar = tqdm(total=len(image_files), desc='Compress Images', unit='img')
    for image_name in image_files:
        pre, ext = os.path.splitext(image_name)
        process_bar.set_description(f"Processing {image_name}")
        
        image_path = os.path.join(image_folder_path, image_name)
        
        # 图片保存路径
        if save_format.lower() == 'retain':
            image_save_path = os.path.join(save_folder_path, pre) + ext
        elif save_format.lower() in ImageFormat:
            image_save_path = os.path.join(save_folder_path, pre) + save_format.lower()
        else:
            raise KeyError(f"❌  The value of save_format is illegal!")
        
        
        log_image_paths.append(image_path)
        log_image_save_paths.append(image_save_path)
        
        _factor = 1
        _save = False
        while True:
            img = Image.open(image_path)
            # 防止图片因为exif信息导致的旋转
            img = ImageOps.exif_transpose(img)  
                
            # 转换图片为RGB模式
            img = img.convert('RGB')
                
            # 获取图片大小（以KB为单位）
            size_kb = get_file_size(fp=image_path, unit='kb')
            log_file_sizes.append(size_kb)
            
            if size_kb > target_size_kb:
                if keep_size:  # 使用高斯模糊
                    img = img.filter(ImageFilter.GaussianBlur(_factor))
                    _factor += 1
                    
                    # 保存图片
                    img.save(image_save_path)
                    image_path = image_save_path
                    _save = True
                else:  # 直接缩小图片尺寸
                    scale_factor = (target_size_kb / size_kb) ** 0.5
                    new_size = tuple(int(dim * scale_factor) for dim in img.size)
                    img.thumbnail(new_size, Image.LANCZOS)
                    img.save(image_save_path)
                    image_path = image_save_path
                    _save = True
            else:
                # 需要保证至少存储了一次
                if not _save:
                    img.save(image_save_path)
                    _save = True
                break
        process_bar.update()
    process_bar.close()
    
    if log:
        ptable = prettytable.PrettyTable(field_names=['SRC', 'DST', 'SIZE (kb)'])
        for src, dst, size in zip(log_image_paths, log_image_save_paths, log_file_sizes):
            ptable.add_row([src, dst, size])
        
        log_sp = os.path.join(save_folder_path, f"log-{get_current_time(time_format='%Y%m%d-%H.%M')}.txt")
        with open(log_sp, 'w') as f:
                f.write(ptable.get_string())
                
        xprint(f"日志已保存在 {log_sp}!", color='blue', bold=True, hl='>')


if __name__ == "__main__":
    compress_images(image_folder_path="Datasets/Web/images", target_size_kb=500, 
                    keep_size=True, save_folder_path='Datasets/Web/images/compress_images', 
                    save_format='retain', log=True, verbose=True)
    