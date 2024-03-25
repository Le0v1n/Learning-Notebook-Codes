import os
import sys
import argparse
import torch
import re
import inspect
from PIL import Image

from typing import Optional, Union
sys.path.append(os.getcwd())

from utils.outer import xprint, print_arguments


def check_function(obj):
    """检查一个函数是什么类型

    Args:
        obj (any): 传入任意参数

    Returns:
        str: 
            'function': 是一个函数
            'callable': 是一个可调用对象
            'variable': 是一个变量
    """
    if inspect.isfunction(obj):
        return 'function'
    elif callable(obj):  # 依次执行，因此排除了函数的可能性
        return 'callable'
    else:
        return 'variable'


def check_keys_in_dict(d: dict, *args, verbose=True) -> Optional[int]:
    """检查一个字典是否有对应的关键字

    Args:
        d (dict): 被检查的字典

    Raises:
        KeyError: 传入的必须是字典

    Returns:
        Optional[int]: 如果有关键字没有，则抛出异常，否则返回None
    """
    assert isinstance(d, dict), f"传入的必须是字典而非 {type(d)}!"
    
    # 获取缺失的关键字
    missing_keys = [key for key in args if key not in d]
    
    if missing_keys:
        raise KeyError(f"以下关键字不存在于字典中: {', '.join(map(repr, missing_keys))}")
    
    else:
        xprint("✔️  数据检查通过", color='green', bold=True) if verbose else ...

    return None
    
    
def process_opt(opt: argparse.Namespace):
    assert isinstance(opt, argparse.Namespace), f"❌  传入的参数数据类型有误 -> {type(opt)}"
    
    # 处理num_workers(train & val)
    nw = min([os.cpu_count(), 8])  # number of workers
    if isinstance(opt.train_num_workers, str):
        if opt.train_num_workers.lower() in ('auto', ):  # 自动计算num workers
            opt.train_num_workers = nw
        else:  # 将其转换为int
            opt.train_num_workers = int(opt.train_num_workers)

        if opt.val_num_workers.lower() in ('auto', ):  # 自动计算num workers
            opt.val_num_workers = nw
        else:  # 将其转换为int
            opt.val_num_workers = int(opt.val_num_workers)
            
    # 处理visualizer_name
    if opt.visualizer_name.lower() in ('tensorboard', 'tb', 'board', 'default'):
        opt.visualizer_name = 'TensorboardVisBackend'
    elif opt.visualizer_name.lower() in ('wandb', 'wb'):
        opt.visualizer_name = 'WandbVisBackend'
    
    # 打印参数
    # screen_clear(clear=True)
    print_arguments(**vars(opt))
    
    
def process_gpu(gpus_id: Union[int, str]):
    """计算GPU索引

    Args:
        gpu (Union[int, str]): 可以输入int也可以输入str。
            如果输入数字，表明要使用几个GPU
            如果输入str表示使用哪些GPU索引 (只能包含数字和逗号)
            e.g.
                gpu=1 --> 使用一个gpu; 
                gpu='1' --> 使用索引值为1的GPU; 
                gpu="0,3,4" --> 使用索引值为0,3,4的GPU

    Returns:
        str: 返回的GPU索引 (like: 0,1,2,3)
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()  # 计算当前GPU的个数
        if isinstance(gpus_id, int):  # 如果是数字
            assert gpus_id <= num_gpus, f"GPU个数设置不正确: 当前设置GPU个数为 {gpus_id}，实际GPU总个数为 {num_gpus}!"
            gpus_id = ",".join(map(str, list(range(gpus_id))))
            return gpus_id
        elif isinstance(gpus_id, str):  # 如果是字符串
            assert re.fullmatch(r'^[\d,]+$', gpus_id), "gpu参数中包含非法字符（只能包含数字和逗号）！"
            _len = len(gpus_id.strip(','))
            assert _len <= num_gpus, f"GPU个数设置不正确: 当前设置GPU个数为 {_len}，实际GPU总个数为 {num_gpus}!"
            return gpus_id
    else:
        xprint(f"⚠️  当前环境没有 GPU！", color='red', bold=True)
        return None
    
    
def process_args_True_False(args):
    for k, v in vars(args).items():
        if v == 'True':
            setattr(args, k, True)
        elif v == 'False':
            setattr(args, k, False)
    return args


def is_folder(input_path: str) -> bool:
    """
    检查给定的一个路径是否是文件夹，如果是则返回True，否则返回False
    Args:
        input_path (str): 给定的一个路径
    Returns:
        bool: 是不是文件夹
    """
    if os.path.isfile(input_path):
            return False
    elif os.path.isdir(input_path):
        return True
    else:
        raise FileNotFoundError(f"❌  路径 [{input_path}] 不存在!")


def get_latest_folder_and_file(dir_path: str, exclude_folder_name=None, exclude_file_name=None, verbose=False) -> list:
    """返回指定目录下最近修改的文件和文件夹路径，排除特定的文件夹和文件名
    Args:
        dir_path (str): 指定目录的路径
        exclude_folder_name (str, optional): 要排除的文件夹名称，默认为None，即不排除任何文件夹
        exclude_file_name (str, optional): 要排除的文件名称，默认为None，即不排除任何文件
        verbose (bool, optional): verbose to terminal. Defaults to False.
    Returns:
        list: [文件路径, 文件夹路径]
    """
    # 获取目录中所有文件和文件夹的名称
    files = [file for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file))]
    folders = [folder for folder in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, folder))]

    # 排除特定的文件夹和文件名
    if exclude_folder_name:
        folders = [folder for folder in folders if exclude_folder_name not in folder]
    if exclude_file_name:
        files = [file for file in files if exclude_file_name not in file]
    if not files and not folders:
        xprint(f"⚠️  {dir_path}中没有文件和文件夹或所有文件和文件夹都被排除!", color='red')
        return None

    # 分别按修改时间排序文件和文件夹
    sorted_files = sorted(files, key=lambda file: os.path.getmtime(os.path.join(dir_path, file)), reverse=True)
    sorted_folders = sorted(folders, key=lambda folder: os.path.getmtime(os.path.join(dir_path, folder)), reverse=True)

    # 获取最新修改的文件和文件夹的名称
    latest_file = sorted_files[0] if sorted_files else None
    latest_folder = sorted_folders[0] if sorted_folders else None
    if verbose:
        _content = ''
        if latest_file:
            _content += f"最新修改的文件名称是  ：{os.path.join(dir_path, latest_file)}\n"
        if latest_folder:
            _content += f"最新修改的文件夹名称是：{os.path.join(dir_path, latest_folder)}"
        if _content:
            xprint(_content, color='blue', hl='>', hl_num=2)

    # 返回最新修改的文件和文件夹的路径
    return [os.path.join(dir_path, latest_file) if latest_file else None,
            os.path.join(dir_path, latest_folder) if latest_folder else None]
    
    
def add_prefix(fp: str, prefix: str) -> str:
    """为路径中的文件名称添加前缀

    - Args:
        - `fp (str)`: 路径
        - `prefix (str)`: 前缀名称

    - Returns:
        - `str`: 返回添加了前缀的路径
    """
    __dir = os.path.dirname(fp)
    __file_name = os.path.basename(fp)
    __prefix, __extension = os.path.splitext(__file_name)
    
    return os.path.join(__dir, prefix + __prefix + __extension)


def add_suffix(fp: str, suffix: str) -> str:
    """为路径中的文件名称添加后缀

    - Args:
        - `fp (str)`: 路径
        - `prefix (str)`: 后缀名称

    - Returns:
        - `str`: 返回添加了后缀的路径
    """
    __dir = os.path.dirname(fp)
    __file_name = os.path.basename(fp)
    __prefix, __extension = os.path.splitext(__file_name)
    
    return os.path.join(__dir, __prefix + suffix + __extension)


def is_rgb_image(file_path: str) -> bool:
    """检查一张图片是不是rgb或rgba图片

    Args:
        file_path (str): 图片的路径

    Returns:
        bool: 是否为rgb或rgba图片
    """
    try:
        # 打开图片
        with Image.open(file_path) as img:
            # 检查图片的模式是否为RGB或RGBA
            return img.mode in ['RGB', 'RGBA']
    except IOError:
        # 如果打开图片时出现错误，返回False
        return False
    
    
def is_pillow_obj(image):
    return hasattr(image, 'mode') and isinstance(image.mode, str)

    
if __name__ == "__main__":
    _gpu = process_gpu(gpus_id='1,2,3,4')
    print(_gpu)
    print(type(_gpu))
    
    path = get_latest_folder_and_file('utils', verbose=True, 
                                      exclude_folder_name=None,
                                      exclude_file_name=None)
    # path = get_latest_folder_and_file('utils', verbose=True, 
    #                                   exclude_folder_name='__pycache__',
    #                                   exclude_file_name='checker.py')
    print(f"{path = }")
    
    