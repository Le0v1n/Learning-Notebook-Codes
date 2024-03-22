import os
import sys
import random
from typing import Union

sys.path.append(os.getcwd())
from utils.outer import xprint
from utils.items import SEG_TASKS


def generator_palette_for_classes(num_classes, task):
    palette = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                for _ in range(num_classes)]

    if task in SEG_TASKS and palette:
        palette[0] = (127,127,127)  # 语义分割的背景

    return palette

                
def create_folder(dst_dir, increment=False, verbose=False) -> str:
    """创建文件夹

    - Args:
        - `dst_dir (str)`: 文件夹路径
        - `increment (bool)`: 是否开启递增文件夹模式. Defaults to False.
            - `increment=True`: 若目标文件夹存在，则创建一个带有后缀的文件夹进行区分，如 runs/train 存在 -> 创建 runs/train2
            - `increment=False`: 若目标文件夹存在，则不再创建
            - 💡  默认为 False
        - `verbose (bool)`: 详细输出. Defaults to False.
    
    - Return:
        - `dst_dir (str)`: 返回最终的文件夹路径
        
    - ⚠️  WARNING：当开启 `increment=True`，请注意接受该函数的返回值，因为 `dst_dir` 已被更新
    """
    assert isinstance(dst_dir, str), f"❌  请传入一个文件夹路径而非 {dst_dir}!"

    # 如果文件夹存在
    if os.path.exists(dst_dir):
        if increment:  # 文件夹递增
            path = Path(dst_dir)
            suffix = 1
            while path.exists():
                suffix += 1
                path = Path(dst_dir + str(suffix))
            dst_dir = str(path)
            xprint(f"⚠️  Folder has existed, create increment folder -> {dst_dir}", color='yellow') if verbose else ...

            # 递归调用自身来创建新的文件夹
            return create_folder(dst_dir, verbose=verbose, increment=increment)
        else:
            xprint(f"⚠️  Folder {dst_dir} has existed!", color='yellow') if verbose else ...

    # 如果文件夹不存在
    else:
        if not os.path.exists(os.path.dirname(dst_dir)):  # 如果父级文件夹不存在，则发出警告
            xprint(f"⚠️  WARNING: The parent folder doesn't exist for {dst_dir}!", color='yellow')
        
        os.makedirs(dst_dir, exist_ok=True)
        xprint(f"✔️  Folder {dst_dir} has been created!", color='yellow') if verbose else ...
    
    return dst_dir


def rgb2hex(rgb_color: Union[tuple, list]) -> str:
    """将RGB颜色转换为HEX格式

    Args:
        rgb_color (Union[tuple, list]): RGB颜色

    Returns:
        str: HEX格式的颜色代码
    """
    return "#{:02x}{:02x}{:02x}".format(rgb_color[0], rgb_color[1], rgb_color[2])


def generator_rgb_colors(num_color: int, return_type: str = 'list', 
                         format_color: str = 'rgb') -> Union[list, dict]:
    """生成RGB颜色列表

    Args:
        num_color (int): 颜色的个数
        return_type (str, optional): 返回的数据类型 (list, dict). Defaults to 'list'.
        format_color (str, optional): 颜色格式 (rgb, hex). Defaults to 'rgb'.

    Returns:
        Union[list, dict]: 返回生成的RGB颜色列表
    """
    # 生成一个包含随机RGB颜色的列表
    colors_rgb = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                  for _ in range(num_color)]

    # 根据format参数决定返回RGB还是HEX格式
    if format_color.lower() == 'hex':
        colors = [rgb2hex(color) for color in colors_rgb]
    elif format_color.lower() in ('rgb', 'default'):
        colors = colors_rgb
    else:  # 默认返回RGB格式
        raise KeyError("Please input correct param of 'format', like 'rgb' or 'hex'!")

    # 根据return_type参数返回列表或字典
    if return_type.lower() in ('list', 'ls', 'lst'):
        return colors
    elif return_type.lower() in ('dict', 'd'):
        return {idx: color for idx, color in enumerate(colors)}
    else:
        raise KeyError("Please input correct param of 'return_type', like 'list' or 'dict'!")
    
    
if __name__ == "__main__":
    ...