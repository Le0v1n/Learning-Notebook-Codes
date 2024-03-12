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

                
def create_folder(fp, exist_ok=True, verbose=False):
    """创建文件夹
        当父级文件夹存在是，调用os.mkdir方法
        当父级文件夹不存在时，调用os.makedirs方法

    Args:
        fp (str): 要创建的文件夹路径
        exist_ok (bool, optional): 建议为True，如果为False，当fp存在时可能会报错. Defaults to True.
    """
    if os.path.exists(fp):  # 如果文件夹存在，则不创建
        xprint(f"⚠️  Folder {fp} has existed!", color='yellow') if verbose else ...
        return
    elif not os.path.exists(os.path.dirname(fp)):  # 如果父级文件夹不存在，则发出警告
        xprint(f"⚠️  The parent folder doesn't exists!", color='yellow')
        os.makedirs(fp, exist_ok=exist_ok)
    else:  # 如果父级文件夹存在且文件夹不存在，那么创建
        xprint(f"✔️  Folder {fp} has been created!") if verbose else ...
        os.mkdir(fp)


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