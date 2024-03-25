import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from PIL import Image
from typing import Union, Optional, List, Tuple, Dict
from tqdm import tqdm

from utils.checker import add_prefix, add_suffix, is_folder, is_rgb_image, is_pillow_obj
from utils.generator import create_folder, generator_rgb_colors
from utils.getter import get_files
from utils.items import ImageFormat
from utils.outer import xprint


def pil_to_opencv(image):
    assert is_pillow_obj(image), f"❌  传入的变量并非 PIL 的图片对象!"
    
    # 转换为RGB排布
    image = image.convert('RGB')
    
    # 转换为OpenCV格式
    opencv_image = np.array(image)
    opencv_image = opencv_image[:, :, ::-1].copy()
    
    return opencv_image


def resize_image_keep_aspect_ratio(image: np.ndarray, size: Union[tuple, list], fill_color: int = 114):
    """调整图像大小并保持宽高比不变，使用填充颜色填充边缘。

    - Args:
        - `image`: 输入图像 (OpenCV 读取得到)
        - `size`: 输出图像的期望大小（宽度，高度）。
        - `fill_color`: 填充颜色值。
            - 默认为 114 (灰色)
            - 💡  数据说明：0 (黑色) ---> 255 (白色)

    - Returns:
        - 调整大小后的图像，保持宽高比并填充边缘。
    """
    if isinstance(size, list):
        size = tuple(size)

    # 获取图像尺寸
    h, w = image.shape[:2]
    target_w, target_h = size

    # 计算宽高比
    aspect_ratio_src = w / h
    aspect_ratio_target = target_w / target_h

    # 计算调整大小的因子
    if aspect_ratio_src > aspect_ratio_target:
        # 基于宽度进行调整
        new_w = target_w
        new_h = int(new_w / aspect_ratio_src)
    else:
        # 基于高度进行调整
        new_h = target_h
        new_w = int(new_h * aspect_ratio_src)

    # 调整图像大小并保持宽高比
    resized_img = cv2.resize(image, (new_w, new_h))

    # 创建具有所需大小的新图像，并用指定颜色填充
    filled_img = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)

    # 计算粘贴调整大小图像的位置
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    # 将调整大小的图像粘贴到填充的图像上
    filled_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

    return filled_img


def gray2pseudo(fp: str, 
                pseudo_save_dir: Union[str, None] = None, 
                num_color: Union[int, None] = 80,
                palette: Dict = None,
                use_tqdm: bool = False,
                verbose: bool = False) -> str:
    """将指定文件夹中或者指定灰度图转换为伪彩色图

    - Args:
        - `fp (str)`: 
            - 可选：灰度图的图片路径
            - 可选：灰度图所在文件夹（会对所有灰度图进行转换）
        - `pseudo_save_dir (str)`: 
            - `pseudo_save_dir = 'xxxxx/xxx'`: 保存伪彩色图到指定文件夹
            - `pseudo_save_dir = None`: 保存伪彩色图到原图的pseudo文件夹下 -> fp/pseudo
        - `num_color (Union[int, None])`: 
            - 类别数量，用来随机生成调色板
            - ⚠️ 与 `palette` 参数互斥
            - 默认为 80
        - `palette` (Tuple[Tuple[int, int, int], Tuple[int, int, int], ...]):
            - 指定的调色板
                - 格式为:
                    - 格式1: `palette = {类别: RGB颜色}` -> `palette = {int: (int, int, int)}`
                    - 格式2: `palette = ((r, g, b), ...)` -> `palette = ((int, int, int), ...)`
                    - 格式3: `palette = [(r, g, b), ...]` -> `palette = [(int, int, int), ...]`
                    - 格式4: `palette = [[r, g, b], ...]` -> `palette = [[int, int, int], ...]`
                    - 格式5: `palette = ([r, g, b], ...)` -> `palette = [[int, int, int], ...]`
                - 举例：`palette = {0: (114, 114, 114), 1: (255, 0, 255), ...}`
            - ⚠️ 与 `num_color` 参数互斥
            - 默认为 None
        - `use_tqdm (bool)`: 
            - 是否使用 tqdm 进度条
            - 默认不使用
        - `verbose (bool)`: 
            - 是否输出详细信息
            - 默认为 False
    
    - Return:
        - str: 伪彩色图所在的文件夹路径
    
    - Notes:
        - `fp`：因为可以是一个文件夹路径，所以请确保文件夹中都是灰度图（不是灰度图的会自动删除掉）
        - `num_color` 并不强制，只要大于类别数就行
            - 每次根据 `num_color` 生成的 palette 都是随机的
            - 随机生成的 palette 的第一个类别强制为 `(114, 114, 114)`，即灰色
        - ⚠️  如果在别处使用了 tdqm，那么 `use_tqdm=True` 可能会报错!
    """
    # ========== 获取灰度图 list (绝对路径) ==========
    if is_folder(fp):  # 如果是文件夹
        grey_list = [os.path.join(fp, file) for file in os.listdir(fp) if file.endswith('.png')]
        fp_dir = fp
    else:  # 如果是文件
        fp_dir = os.path.dirname(fp)  # 所在文件夹路径
        # 判断是否为png格式
        if os.path.basename(fp).endswith('.png'):  
            grey_list = [fp]
        else:
            raise ValueError(f"❌ 输入有误!\n\t"
                             f"1. 请输入灰度图路径或者灰度图所在文件夹路径\n\t"
                             f"2. 请确保灰度图后缀为 '.png'，当前为 [{os.path.splitext(fp)[1]}]")
            
    # 对得到的grey_images_list进行判断，删除不是灰度图的元素
    for image_path in grey_list:
        if is_rgb_image(image_path):  # 不是灰度图
            grey_list.remove(image_path)  # 删除不是灰度图的图片路径
    
    assert len(image_path) > 0, f"❌ 没有读取到灰度图！"
    
    # 创建 tqdm 进度条
    process_bar = tqdm(total=len(grey_list), desc="grey2pseudo", unit='png') if use_tqdm else None
    
    # 处理 palette
    if num_color and palette:  # 如果二者同时存在，则报错！
        raise ValueError(f"❌ 'num_classes' 和 'palette' 参数是互质的，请确保其中有一个是 None!")
    
    if num_color:  #  生成随机的 palette
        color_map = generator_rgb_colors(num_color=num_color, return_type='dict')  # 随机定义一个 RGB 颜色字典
        color_map[0] = (114, 114, 114)  # 修改 'background' 类别的颜色
    else:  # 使用指定的调色板
        if isinstance(palette, dict):
            color_map = palette
        
        elif isinstance(palette, (list, tuple)):
            color_map = {idx: tuple(color) for idx, color in enumerate(palette)}
        else:
             raise ValueError(f"❌ 请确保 'palette' 参数格式正确!\n\t"
                              f"当前为: {palette}")
             
    # 创建保存 pseudo 文件夹
    if not pseudo_save_dir or pseudo_save_dir == fp_dir:  # 如果 pseudo_save_dir is None or pseudo_save_dir与fp_dir相同
        pseudo_save_dir = os.path.join(fp_dir, 'pseudo')
        
    create_folder(pseudo_save_dir, verbose=verbose)
    
    # 遍历灰度图并进行转换
    for grey_path in grey_list:
        process_bar.set_description(f"Processing {os.path.basename(grey_path)}") if use_tqdm else ...
        
        # 定义伪彩色图保存路径
        pseudo_save_path = os.path.join(pseudo_save_dir, os.path.basename(grey_path))
        
        # 读取灰度图
        gray_img = cv2.imread(grey_path, cv2.IMREAD_GRAYSCALE)
        
        # 创建一个大小相同、空的彩色图
        pseudo_color_img = np.zeros((gray_img.shape[0], gray_img.shape[1], 3), dtype=np.uint8)
        
        # 给每一个灰度像素匹配颜色
        for gray_level, color in color_map.items():
            pseudo_color_img[gray_img == gray_level] = color
        
        # 保存伪彩色图
        Image.fromarray(pseudo_color_img).save(pseudo_save_path)
        
        process_bar.update() if use_tqdm else ...
    process_bar.close() if use_tqdm else ...
    
    xprint(f"灰度图转伪彩色图已完成，保存在 {pseudo_save_dir}", 
           color='blue', hl='>', bold=True) if verbose else ...
    
    return pseudo_save_dir


def image_interpolate_as_pytorch(image: np.ndarray, 
                                 target_size: Union[int, tuple]=None,
                                 scale_factor: Union[float, tuple]=None,
                                 mode: str = 'bilinear',
                                 align_corners: Union[bool, None] = None):
    """将 OpenCV 对象 (PIL 对象) 按照 PyTorch 的方式进行插值

    - Args:
        - `input (np.ndarray)`: OpenCV 的图片对象
        - `target_size (Union[int, tuple], optional)`: 
            - 目标尺寸
            - 默认为 None.
            - ⚠️  与 scale_factor 互斥
        - `target_size (Union[float, tuple], optional)`: 
            - 按百分比缩放
            - 默认为 None.
            - ⚠️  与 target_size 互斥
        - `mode (str)`: 插值方法
            - 💡  默认为 `'bilinear'`
            - `nearest`: 最近邻插值，这是最简单的方法，新图像中的像素值直接从原始图像中最近的像素获取。
            - `linear`: 线性插值（也称为双线性插值），它在两个方向上分别进行一次线性插值。
            - `bilinear`: 双线性插值，这是对二维图像进行放大或缩小时常用的方法。
            - `bicubic`: 双三次插值，这是一种更复杂的插值方法，可以产生更平滑的边缘。
            - `trilinear`: 三线性插值，用于三维数据。
            - `area`: 区域插值，也称为像素面积关系插值，它通过平均像素区域来计算新像素值。
            - `nearest-exact`: 最近邻插值，但是会保持输入和输出张量的比例关系。
            - `lanczos`: Lanczos 插值，这是一种高质量的插值方法，尤其擅长保留图像细节。
        - `align_corners (Union[bool, None])`: 是否对齐角点
            - 默认为 None.

    - Returns:
        np.ndarray: 返回插值后的 OpenCV 对象
    """
    # 如果是 PIL 对象则转换为 OpenCV 对象
    if is_pillow_obj(image):
        image = pil_to_opencv(image)
    
    # 记录输入的数据类型
    data_type = image.dtype
    
    # 将图像转换为Tensor
    input_tensor = torch.from_numpy(image).float()
    
    # 判断是不是灰度图
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(-1)  # 增加 channel 维度
    
    input_tensor = input_tensor.permute(2, 0, 1)  # 将通道提到前面
    input_tensor = input_tensor.unsqueeze(0)  # 增加 Batch 维度

    # 开始插值
    result = F.interpolate(input=input_tensor, 
                           size=target_size, 
                           scale_factor=scale_factor, 
                           mode=mode, 
                           align_corners=align_corners)

    result = result.squeeze(0)  # 去掉 Batch 维度
    result = result.permute(1, 2, 0)  # 将 Channel 放到最后: [C, H, W] -> [H, W, C]
    result = result.numpy()  # 将Tensor转换为ndarray对象
    result = np.clip(result, 0, 255)  # 先将值限制在 0 到 255 之间
    result = result.astype(data_type)  # 恢复为之前的数据类型
    
    return result


def merge_image_and_pseudo(image_fp: str, 
                           pseudo_fp: str,
                           save_dir: Union[str, None] = None, 
                           blend_factor: float = 0.3, 
                           result_extension: str = 'auto', 
                           use_tqdm: bool = False,
                           interpolate_size: str = 'label',
                           interpolate_mode: str = 'bilinear',
                           verbose: bool = False) -> str:
    """将图片和伪彩色图进行融合

    - Args:
        - `image_fp (str)`: 图片路径
            - 可选：图片路径 (file_path)
            - 可选：图片所在文件夹路径 (dir)
        - `pseudo_fp (str)`: 伪彩色图路径
            - 可选：伪彩色图路径 (file_path)
            - 可选：伪彩色图所在文件夹路径 (dir)
        - `save_dir (Union[str, None], optional)`: 结果保存路径
            - 可选：结果保存文件夹路径 (dir)
            - 可选：None -> image_fp/merged
            - 默认为 `None`
        - `blend_factor (float, optional)`: 融合系数
            - 默认为 `0.3`
        - `result_extension (str, optional)`: 融合后图片的后缀
            - 特定的图片格式：所有融合后的图片均保存为该格式
            - `'auto'`: 保留原有的图片格式
            - `None`: 保留原有的图片格式
            - 默认为 `'auto'`.
        - `use_tqdm (bool, optional)`: 是否使用 tqdm 进度条
            - 默认为 `False`
        - `interpolate_size (str)`: 插值为谁的尺寸
            - 默认为 `'label'`: 如果大小不一致，图片先 插值 为 label 的尺寸 -> 融合 -> 上采样到原图大小
            - 可选 `'image'`: 如果大小不一致，label 先 插值 为 image 的尺寸 -> 融合
        - `interpolate_mode (str)`: 插值方法
            - 默认为 `'bilinear'`
            - `nearest`: 最近邻插值，这是最简单的方法，新图像中的像素值直接从原始图像中最近的像素获取。
            - `linear`: 线性插值（也称为双线性插值），它在两个方向上分别进行一次线性插值。
            - `bilinear`: 双线性插值，这是对二维图像进行放大或缩小时常用的方法。
            - `bicubic`: 双三次插值，这是一种更复杂的插值方法，可以产生更平滑的边缘。
            - `trilinear`: 三线性插值，用于三维数据。
            - `area`: 区域插值，也称为像素面积关系插值，它通过平均像素区域来计算新像素值。
            - `nearest-exact`: 最近邻插值，但是会保持输入和输出张量的比例关系。
            - `lanczos`: Lanczos 插值，这是一种高质量的插值方法，尤其擅长保留图像细节。

    - Returns:
        - `str`: 返回融合后图片所在文件夹路径
        
    - Notes:
        - ⚠️  如果在别处使用了 tdqm，那么 `use_tqdm=True` 可能会报错!
    """
        
    # 处理 image_fp
    if is_folder(image_fp):  # 文件夹
        images_list = get_files(fp=image_fp, extension='image', path_style='relative')  # 使用相对路径
        image_root = image_fp  # 图片所在文件夹路径
    else:  # 文件
        if os.path.splitext(image_fp)[-1].lower() in ImageFormat:
            images_list = [image_fp]
        else:
            raise ValueError(f"❌  当前文件 ({image_fp}) 后缀并非为常见图片格式")
        image_root = os.path.dirname(image_fp)  # 图片所在文件夹路径
    
    # 处理 result_extension
    result_extension = result_extension.lower()
    if not result_extension or result_extension not in ('auto', 'no_change'):
        assert result_extension in ImageFormat, f"❌  参数 result_extension={result_extension} 并不是图片格式！"

    # 创建 tqdm 对象
    process_bar = tqdm(total=len(images_list), 
                       desc=f"Merge image & pseudo with {blend_factor}") if use_tqdm else None
    
    # 处理 save_dir
    if not save_dir:  # 如果 save_dir is None
        save_dir = os.path.join(image_root, 'merged')
    create_folder(save_dir, verbose=verbose)
    
    # 遍历原图像列表
    for idx, image_path in enumerate(images_list):
        image_name = os.path.basename(image_path)
        process_bar.set_description(f"Processing {image_name}") if use_tqdm else ...
        
        # 确定路径
        pre, ext = os.path.splitext(image_name)  # 获取前缀和后缀
        pseudo_path = os.path.join(pseudo_fp, pre) + '.png'
        
        assert os.path.exists(image_path), f"❌  {image_path} doesn't exist!"
        assert os.path.exists(pseudo_path), f"❌  {pseudo_path} doesn't exist!"
            
        # 加载原图像和 pseudo
        original_image = cv2.imread(image_path)
        pseudo_image = cv2.imread(pseudo_path)

        # 调整 pseudo 图像的大小以匹配原图像
        if original_image.shape[:2] != pseudo_image.shape[:2]:  # image 和 label 的尺寸不一致
            if interpolate_size.lower() in ('image'):  # 将 label 插值为和 image 相同的尺寸
                xprint(f"将 label 插值为和 image 相同的尺寸") if verbose else ...
                pseudo_image = image_interpolate_as_pytorch(pseudo_image, 
                                                            target_size=original_image.shape[:2],
                                                            mode=interpolate_mode)
                
            elif interpolate_size.lower() in ('label'):  # 将 image 插值为和 label 相同的尺寸
                xprint(f"将 image 插值为和 label 相同的尺寸") if verbose else ...
                original_image = image_interpolate_as_pytorch(original_image, 
                                                              target_size=pseudo_image.shape[:2],
                                                              mode=interpolate_mode)
                
            else:
                raise KeyError(f"❌ 请给出正确的 interpolate_size 参数 ('image' or 'label')\n\t"
                               f"当前为 {interpolate_size}!")

        else:  # 无需调整
            print(f"image 和 label 的尺寸正好，无需调整") if verbose else ...
            
        # 将原图像和 pseudo 图像进行融合
        merged_image = cv2.addWeighted(src1=original_image, 
                                       alpha=blend_factor, 
                                       src2=pseudo_image, 
                                       beta=1 - blend_factor,
                                       gamma=0)
        
        # 获取输出文件名
        if result_extension not in ('auto', 'no_change'):  # 替换为统一的格式
            result_path = os.path.join(save_dir, pre + result_extension)
        else:  # 保留原本的格式
            result_path = os.path.join(save_dir, pre + ext)

        # 保存融合后的图像
        cv2.imwrite(result_path, merged_image)
        
        # 更新 tqdm
        process_bar.update() if use_tqdm else ...
    process_bar.close() if use_tqdm else ...

    xprint(f"✔️  融合图像已经完成，保存文件夹路径为: {save_dir}", 
           color='green', hl='>', hl_num=2, hl_style='full') if verbose else ...
