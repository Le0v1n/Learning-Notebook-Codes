import os
import sys
import cv2
from typing import Union

sys.path.append(os.getcwd())
from utils.outer import xprint
from utils.getter import get_files
from utils.checker import is_folder, add_prefix, add_suffix
from utils.items import ImageFormat
from utils.generator import create_folder

try:
    from tqdm.rich import tqdm
except:
    from tqdm import tqdm


def merge_image_and_pseudo(image_path_or_dir: str, 
                           pseudo_dir: Union[str, None] = None,
                           save_dir: Union[str, None] = None, 
                           pseudo_prefix: Union[str, None]=None,
                           pseudo_suffix: Union[str, None]=None,
                           blend_factor: float = 0.3, 
                           result_extension: str = 'auto', 
                           use_tqdm: bool = False,
                           verbose: bool = False) -> str:
    """将图片和伪彩色图进行融合

    - Args:
        - `image_path (str)`: 图片路径或所在文件夹路径
        - `pseudo_dir (Union[str, None], optional)`: 伪彩色图所在文件夹路径
            - 可以选择特定的文件夹路径
            - 如果为 `None`，则使用 `image_path` 文件夹路径
            - 💡  默认为 `None`.
        - `save_dir (Union[str, None], optional)`: 输出文件夹路径
            - 可以选择特定的文件夹路径
            - 如果为 `None`，则使用 `image_path` 文件夹路径
            - 💡  默认为 `None`.
        - `pseudo_prefix (Union[str, None], optional)`: 伪彩色图是否有前缀
            - 假如图片的名称为 xxx.jpg，但伪彩色图有前缀，如 pseudo-xxx.png，我们需要填写该参数
            - 如果为 `None`，则不添加后缀
            - 💡  默认为 `None`.
        - `pseudo_suffix (Union[str, None], optional)`: 伪彩色图是否有后缀
            - 假如图片的名称为 xxx.jpg，但伪彩色图有后缀，如 xxx-pseudo.png，我们需要填写该参数
            - 如果为 `None`，则不添加后缀
            - 💡  默认为 `None`.
        - `blend_factor (float, optional)`: 融合系数
            - 💡  默认为 `0.3`
        - `result_extension (str, optional)`: 融合后图片的后缀
            - 特定的图片格式：所有融合后的图片均保存为该格式
            - `'auto'`: 保留原有的图片格式
            - `None`: 保留原有的图片格式
            - 💡  默认为 `'auto'`.
        - `use_tqdm (bool, optional)`: 是否使用 tqdm 进度条
            - 💡  默认为 `False`

    - Returns:
        - `str`: 返回融合后图片所在文件夹路径
        
    - Notes:
        - ⚠️  如果在别处使用了 tdqm，那么 `use_tqdm=True` 可能会报错!
    """
        
    # 如果输入的是文件夹路径，则获取文件夹中的所有图片文件
    if is_folder(image_path_or_dir):
        image_dir = image_path_or_dir  # 图片所在文件夹路径
        images_list = get_files(fp=image_path_or_dir, extension='image', path_style='relative')  # 使用相对路径
    else:
        image_dir = os.path.dirname(image_path_or_dir)  # 图片所在文件夹路径
        if os.path.splitext(image_path_or_dir)[-1].lower() in ImageFormat:
            images_list = [image_path_or_dir]
        else:
            raise ValueError(f"❌  当前文件后缀 ({os.path.splitext(image_path_or_dir)[-1]}) 并非为常见图片格式")
    
    if not save_dir:
        save_dir = image_dir
    
    result_extension = result_extension.lower()
    if not result_extension or result_extension not in ('auto', 'no_change'):
        assert result_extension in ImageFormat, f"❌  参数 result_extension={result_extension} 并不是图片格式！"
        
    # 创建保存文件夹
    create_folder(save_dir, verbose=verbose)

    # 创建 tqdm 对象
    process_bar = tqdm(total=len(images_list), 
                       desc=f"Merge image & pseudo with {blend_factor}") if use_tqdm else None
    
    # 遍历原图像列表
    for idx, image_path_or_dir in enumerate(images_list):
        image_name = os.path.basename(image_path_or_dir)
        process_bar.set_description(f"Processing {image_name}") if use_tqdm else ...
        
        # 确定路径
        pre, ext = os.path.splitext(image_name)  # 获取前缀和后缀
        
        if pseudo_dir:  # 使用指定路径的伪彩色图
            pseudo_path = os.path.join(pseudo_dir, pre) + '.png'
        else:  # 使用 image 目录下的伪彩色图
            pseudo_path = os.path.join(image_dir, pre) + '.png'
            
        # 处理前缀和后缀
        pseudo_path = add_prefix(pseudo_path, pseudo_prefix) if pseudo_prefix else pseudo_path
        pseudo_path = add_suffix(pseudo_path, pseudo_suffix) if pseudo_suffix else pseudo_path
            
        # 加载原图像和 pseudo
        original_image = cv2.imread(image_path_or_dir)
        pseudo_image = cv2.imread(pseudo_path)

        # 调整 pseudo 图像的大小以匹配原图像
        if original_image.shape[:2] != pseudo_image.shape[:2]:
            pseudo_image = cv2.resize(pseudo_image, (original_image.shape[1], original_image.shape[0]))

        # 将原图像和 pseudo 图像进行融合
        merged_image = cv2.addWeighted(src1=original_image, 
                                       alpha=blend_factor, 
                                       src2=pseudo_image, 
                                       beta=1 - blend_factor,
                                       gamma=0)

        if not save_dir:  # 使用 image 文件夹路径
            save_dir = image_dir
        
        # 获取输出文件名
        if not result_extension or result_extension not in ('auto', 'no_change'):  # 替换为统一的格式
            result_path = os.path.join(save_dir, pre + '-merge' + result_extension)
        else:  # 保留原本的格式
            result_path = os.path.join(save_dir, pre + '-merge' + ext)

        # 保存融合后的图像
        cv2.imwrite(result_path, merged_image)
        
        # 更新 tqdm
        process_bar.update() if use_tqdm else ...
    process_bar.close() if use_tqdm else ...

    xprint(f"✔️  融合图像已经完成，保存文件夹路径为: {save_dir}", 
           color='green', hl='>', hl_num=2, hl_style='full') if verbose else ...


if __name__ == "__main__":
    merge_image_and_pseudo(
        image_path_or_dir='../datasets/example/images', 
        pseudo_dir='save_dir/segmentation/onnx_infer',
        save_dir='save_dir/segmentation/onnx_infer/merged',
        pseudo_prefix=None,
        pseudo_suffix='-label-pseudo',
        blend_factor=0.3,
        result_extension='auto',
        use_tqdm=True,
        verbose=True
    )