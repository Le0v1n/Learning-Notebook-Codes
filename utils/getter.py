import os
import sys
from datetime import datetime
import inspect
import logging
from typing import Union

sys.path.append(os.getcwd())
from utils.generator import create_folder


def get_files(fp: str, extension: str = 'any', path_style: Union[str, None] = None):
    """获取某一个文件夹下的指定格式的所有文件的路径

    - Args:
        - `fp (str)`: 文件夹路径
        - `extension (str, optional)`: 文件格式，可以是某一格式（`.jpg`），也可以是一个 tuple，
            也可以使用内置的关键字。
            - `'image'`：('.png', '.jpg', '.jpeg', '.gif', '.bmp')
            - `'annotation'`：('.xml', '.csv', '.json', '.txt', '.png')
            - `'any'`：不限制后缀
            - `'all'`：不限制后缀
            - `'whatever'`：不限制后缀
            - `None`：不限制后缀
            - 💡  默认为 `'image'`.
        - `path_style (str)`: 为文件名添加路径
            - `None`：不添加路径
            - `relative`：添加相对路径
            - `absolute`：添加绝对路径
            - 💡  默认为 None -> 不添加路径

    - Returns:
        - `list`: 一个包含所有指定文件的 list
    """
    
    extensions = {
        'image': ('.png', '.jpg', '.jpeg', '.gif', '.bmp'),
        'annotation': ('.xml', '.csv', '.json', '.txt', '.png'),
        'any': None,
        'whatever': None,
        'all': None
    }
    
    assert isinstance(extension, (str, list, tuple, None)), f"请输入正确的文件格式，当前输入的文件格式为 {extension}"
    
    # 检查extension是否为合法关键字
    if isinstance(extension, str):  # 如果是字符串
        if extension.lower() in extensions:
            extension = extensions[extension.lower()]
    else:  # list 或者 tuple
        extension = tuple(extension)
    
    # 文件搜索
    if extension:  # 指定后缀
        files_list = [file for file in os.listdir(fp) if file.endswith(extension)]
    else:  # 任意后缀都行
        files_list = [file for file in os.listdir(fp)]

    # 处理path_style
    if path_style and isinstance(path_style, str):
        if path_style.lower() in ('rel', 'rela', 'relative', 'relate'):
            files_list = [os.path.join(fp, file_name) for file_name in files_list]
        elif path_style.lower() in ('abs', 'absolute'):
            files_list = [os.path.abspath(file_name) for file_name in files_list]
    
    return files_list


def get_file_size(fp, unit='MB', ndigits=4):
    """获取文件大小
    Args:
        fp (str): 文件路径
        unit (str): 单位选项，可以是'KB', 'MB', 'GB'等
        ndigits (int): 小数点后保留的位数
    Returns:
        float: 文件大小(默认为MB)
    """
    
    # 获取文件大小（字节）
    file_size_bytes = os.path.getsize(fp)
    unit = unit.upper()
    
    # 单位到字节倍数的映射
    unit_multipliers = {
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024,
    }
    
    # 根据单位转换文件大小
    if unit in unit_multipliers:
        multiplier = unit_multipliers[unit]
        file_size = round(file_size_bytes / multiplier, ndigits=ndigits)
    else:
        # 默认或未知单位时使用MB
        file_size = round(file_size_bytes / (1024 * 1024), ndigits=ndigits)
        unit = 'MB'
    return file_size
        

def get_current_time(time_format='default') -> str:
    """获取当前时间，并按照指定格式返回。

    参数:
    format_key (str): 时间格式的键。可以是 'default', 'format1'，也可以直接传入%Y%m%d-%H%M%S。
        'default': '%Y%m%d-%H%M%S',  # 年月日时分秒 --> 20240226-102028
        'format1': '%y%m%d-%H%M%S',  # 年月日时分秒 --> 240226-102515
        'format2': '%Y-%m-%d %H:%M:%S',  # 年-月-日 时:分:秒 --> 2024-02-26 10:24:26 

    返回:
    str: 按照指定格式格式化的当前时间。
    """
    time_formats = {
        'default': '%Y%m%d-%H%M%S',  # 年月日时分秒 --> 20240226-102028
        'format1': '%y%m%d-%H%M%S',  # 年月日时分秒 --> 240226-102515
        'format2': '%Y-%m-%d %H:%M:%S',  # 年-月-日 时:分:秒 --> 2024-02-26 10:24:26 
    }
    
    # 获取当前时间
    current_time = datetime.now()
    
    if time_format.lower() in time_formats:
        return current_time.strftime(time_formats[time_format.lower()])
    else:
        return current_time.strftime(time_format)


def get_logger(log_save_path: str = None, verbose: bool = False) -> logging.RootLogger:
    """获取logger实例化对象

    Args:
        log_save_path (str, optional): logger保存的路径. Defaults to None.
        verbose (bool, optional): 是否在终端显示日志. Defaults to False.

    Returns:
        logging.RootLogger: logger实例化对象
    """
    # 获取调用get_logger()函数的信息
    current_frame = inspect.stack()[1]  # 获取调用栈中的当前帧
    caller_file_path = current_frame.filename  # 💡 获取当前帧的文件名
    caller_file_base_path = os.path.basename(caller_file_path) # 💡 获取当前帧的文件名
    caller_function_name = current_frame.function  # 💡 获取当前帧的函数名
    logger_name = f"Logging-{caller_file_base_path}-{caller_function_name}-{get_current_time('%Y%m%d_%H%M%S')}.log"
    
    if not log_save_path:  # 如果没有传入日志的保存路径
        log_save_path = os.path.join(os.path.dirname(caller_file_path), 'local-logs', logger_name)
    else:  # 如果传入了日志的保存路径
        assert isinstance(log_save_path, str), f"❌  log_save_path接收了错误的参数: {log_save_path}({type(log_save_path)})!"
        log_save_path = os.path.join(log_save_path, 'local-logs', logger_name)

    create_folder(fp=os.path.dirname(log_save_path))

    logging.basicConfig(filename=log_save_path, 
                        level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建日志记录器
    logger = logging.getLogger()

    if verbose:
        # 创建控制台处理器并添加到日志记录器
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)
        
    return logger


def get_logger_save_path(logger: logging.RootLogger, relative=True) -> str:
    """返回logger文件的保存路径(相对路径)

    Args:
        logger (logging.RootLogger): logging对象
        relative (bool, optional): 是否返回相对路径 (False则返回绝对路径). Defaults to True.

    Returns:
        str: 返回logger的文件路径
    """
    lsp = logger.handlers[0].baseFilename  # logging_save_path
    
    if relative:
        lsp = os.path.relpath(lsp, os.getcwd())
    return lsp


if __name__ == "__main__":
    # 使用默认格式获取当前时间
    print(get_current_time())
    print(get_current_time('format1'))
    print(get_current_time('format2'))
    print(get_current_time('format3'))
    print(get_current_time('format3'))
    print(get_current_time('%Y%m%d-%H%M'))
    
    print(get_files(fp='Datasets/Web/images/compress_images', path_style='rel'))