import os
import sys
import cv2
import numpy as np
from PIL import Image

sys.path.append(os.getcwd())

from utils.outer import xprint
from utils.generator import create_folder, generator_rgb_colors
from utils.checker import is_folder

try:
    from tqdm.rich import tqdm
except:
    from tqdm import tqdm
    

def gray2pseudo(grey_images_path: str, 
                pseudo_images_save_dir: str, 
                num_color: int = 80,
                use_tqdm: bool = False,
                verbose: bool = False) -> str:
    """将灰度图转换为伪彩色图

    - Args:
        - `grey_images_path (str)`: 灰度图的路径
            - 可选：图片路径或图片所在文件夹
        - `pseudo_images_save_dir (str)`: 保存伪彩色图的文件夹路径
            - 可选：伪彩色图路径或所在文件夹
        - `num_color (int, optional)`: 类别数量. Defaults to 80.
        - `use_tqdm (bool)`: 是否使用 tqdm 进度条. Defaults to False.
        - `verbose (bool)`: Defaults to False.
    
    - Return:
        - str: 伪彩色图所在的文件夹路径
    
    - Notes:
        - `num_color` 并不强制，只要大于类别数就行
        - 每次颜色都是随机的
        - 第一个类别强制为 `(114, 114, 114)` 灰色
        - ⚠️  如果在别处使用了 tdqm，那么 `use_tqdm=True` 可能会报错!
    """
    # ========== 获取灰度图 list (绝对路径) ==========
    if is_folder(grey_images_path):  # 如果是文件夹
        grey_images_list = [os.path.join(grey_images_path, file) for file in os.listdir(grey_images_path) 
                            if file.endswith('.png')]
    else:
        if os.path.basename(grey_images_path).endswith('.png'):
            grey_images_list = [grey_images_path]
        else:
            raise ValueError(f"❌  输入有误!\n\t1. 请输入灰度图路径或者灰度图所在文件夹路径"
                             f"\n\t2. 请确保灰度图后缀为 [.png]，当前为 [{os.path.splitext(grey_images_path)[1]}]")
    
    # 创建 tqdm 进度条
    process_bar = tqdm(total=len(grey_images_list), desc="grey2pseudo", unit='png') if use_tqdm else None
    
    # 创建保存 pseudo 文件夹
    create_folder(pseudo_images_save_dir, verbose=verbose)
    
    # 随机定义一个 RGB 颜色字典
    color_map = generator_rgb_colors(num_color=num_color, return_type='dict')
    color_map[0] = (114, 114, 114)  # 修改 'background' 类别的颜色
    
    # 遍历灰度图并进行转换
    for grey_image_path in grey_images_list:
        process_bar.set_description(f"Processing {os.path.basename(grey_image_path)}") if use_tqdm else ...
        
        # 定义伪彩色图保存路径
        pre, ext = os.path.splitext(os.path.basename(grey_image_path))
        pseudo_save_path = os.path.join(pseudo_images_save_dir, pre + '-pseudo') + ext
        
        # 读取灰度图
        gray_img = cv2.imread(grey_image_path, cv2.IMREAD_GRAYSCALE)
        
        # 创建一个大小相同、空的彩色图
        pseudo_color_img = np.zeros((gray_img.shape[0], gray_img.shape[1], 3), dtype=np.uint8)
        
        # 给每一个灰度像素匹配颜色
        for gray_level, color in color_map.items():
            pseudo_color_img[gray_img == gray_level] = color
        
        # 保存伪彩色图
        Image.fromarray(pseudo_color_img).save(pseudo_save_path)
        process_bar.update() if use_tqdm else ...
    process_bar.close() if use_tqdm else ...
    
    xprint(f"灰度图转伪彩色图已完成，保存在 {pseudo_images_save_dir}", 
           color='blue', hl='>', bold=True) if verbose else ...
    
    return pseudo_images_save_dir


if __name__ == '__main__':
    gray2pseudo(
        grey_images_path='../datasets/exp-dataset/annotations',
        pseudo_images_save_dir='../datasets/exp-dataset/annotations-pseudo', 
        num_color=2)
