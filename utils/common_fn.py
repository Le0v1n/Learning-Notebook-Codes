import os
import sys
import shutil
import prettytable
import re
import random
from datetime import datetime
import pprint as _pprint
import inspect
import logging


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


def print_arguments(*args, **kwargs) -> prettytable.prettytable.PrettyTable:
    """打印参数的函数
    
    *args: 直接传参 --> print_arguments(param1, param2, ...)
    **kwargs: 有关键字的传参 --> print_arguments(key1=param1, key2=param2)
    
    💡Node: 
        1. params_dict=[[param1, param2], ...]则会自动打印
        2. wait=True或confirm=True或check_True或check_params，则会等待用户输入yes后程序会继续执行，目的是检查参数是否正确
        3. show_type=True: 则会显示传参时的参数类型
    """
    table = prettytable.PrettyTable(["index", "type", "name", "value"])
    table.border = kwargs.get("table_border", True)
    table.align["index"] = 'c'
    table.align["type"] = 'c'
    table.align["name"] = 'l'
    table.align["value"] = 'l'
    
    # 添加*args参数
    for arg in args:
        table.add_row([f"{len(table.rows)+1}", type(arg), "", arg])
        
    # 解决 params_dict=[[param1, param2], ...]
    params_dict = kwargs.get('params_dict', None)
    if isinstance(params_dict, list):  # 判断是不是list
        for p in params_dict:
            table.add_row([f"{len(table.rows)+1}", type(p)] + p)
        del kwargs['params_dict']
    
    # 添加**kwargs
    for k, v in kwargs.items():
        table.add_row([len(table.rows)+1, type(v), k, v])
    
    # 何时显示type列
    if not kwargs.get("show_type", False):
        table.del_column(fieldname='type')
        
    # 删除wait和confirm行
    if kwargs.get('wait', False) or kwargs.get('confirm', False) or \
        kwargs.get('check', False) or kwargs.get('check_params', False):
        table.del_row(-1)

    # 不打印
    if not kwargs.get('silent', False):
        print(table)

        # 检查检查，等待用户输入
        if kwargs.get('wait', False) or kwargs.get('confirm', False) or \
            kwargs.get('check', False) or kwargs.get('check_params', False):
            user_input = input("\033[1;31mContinue (Yes/Y)?  \033[0m").lower()
            if user_input in ['yes', 'y']:
                pass
            elif user_input == 'no' or user_input == 'n':
                sys.exit("User exit!\n")
            else:
                print("Invalid input!")
                sys.exit("User exit!")
            
    return table.get_string()
                
                
def create_folder(fp, exist_ok=True, verbose=False):
    """创建文件夹
        当父级文件夹存在是，调用os.mkdir方法
        当父级文件夹不存在时，调用os.makedirs方法

    Args:
        fp (str): 要创建的文件夹路径
        exist_ok (bool, optional): 建议为True，如果为False，当fp存在时可能会报错. Defaults to True.
    """
    if os.path.exists(fp):
        xprint(f"⚠️  {fp} has existed!", color='yellow') if verbose else ...
        return
        
    # get parent folder path
    parent_fp = os.path.dirname(fp)
    
    if not os.path.exists(parent_fp):
        print(f"⚠️  The parent folder doesn't exists!")
        os.makedirs(fp, exist_ok=exist_ok)
    else:
        os.mkdir(fp)
    xprint(f"✔️  Creating {fp} has finished!") if verbose else ...
        
def get_files(fp: str, file_format='image'):
    """获取某一个文件夹下的指定格式的所有文件

    Args:
        fp (str): 文件夹路径
        file_format (str, optional): 文件格式，可以是某一格式（.jpg），也可以是一个tuple，也可以使用内置的关键字（image）. 
                                    Defaults to 'image'.

    Returns:
        list: 一个包含所有指定文件的list 
    """
    if file_format.lower() in ('image', 'images', 'picture', 'pictures', 'pic', 'photo', 'photos'):
        file_format = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    elif file_format.lower() in ('annotation', 'annotations', 'label', 'labels'):
        file_format = ('.xml', '.csv', '.json', '.txt')

    files = [f for f in os.listdir(fp) if f.lower().endswith(file_format)]
    return files


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


def find_text_place_length(text):
    """找到一段文字中的长度：英文1位，中文2位

    Args:
        text (str): 文字

    Returns:
        int: 长度
    """
    pattern_alpha = re.compile(r'[a-zA-Z]')  # 匹配数字和英文字母的正则表达式
    pattern_digit = re.compile(r'\d')  # 匹配数字的正则表达式
    pattern_punct = re.compile(r'[.,;:!?()"]')  # 匹配英文标点符号的正则表达式
    
    alpha_matches = len(pattern_alpha.findall(text))  # 找到所有英文字母的匹配项
    digit_matches = len(pattern_digit.findall(text))  # 找到所有数字的匹配项并计算数量
    punct_matches = len(pattern_punct.findall(text))  # 找到所有英文标点符号的匹配项
    
    return 2 * len(text) - alpha_matches - digit_matches - punct_matches


def screen_clear(clear=False):
    if os.name == 'nt':  # 如果是Windows系统
        os.system('cls')
    else:  # 如果是类Unix系统
        os.system('clear')


def xprint(content:str, color=None, bg_color=None, underline=False, bold=False, end='\n', 
           hl='', hl_style='paragraph', hl_num=1, 
           clear=False, pprint=False):
    """自用的print方法

    Args:
        content (str): 想要print的文字
        color (str, optional): red/random. Defaults to None.
        bg_color (str, optional): red/random. Defaults to None.
        underline (bool, optional): 是否使用下划线. Defaults to False.
        bold (bool, optional): 是否使用粗体. Defaults to False.
        end (str, optional): 结尾. Defaults to '\n'.
        horizontal_line (str, optional): 使用哪种水平线 (- = > < . _). Defaults to ''.
        horizontal_line_length (str, optional): 水平线的长度 (full / paragraph). Defaults to 'paragraph'.
        horizontal_line_num (int): 水平线的个数 (1 / 2). Defaults to 1.
        clear (bool, optional): 是否在打印前清空终端
        pprint (bool, optional): 如果打印的内容不是字符串，且pprint=True，则使用pprint进行打印
    """
    # 定义 ANSI 转义码
    font_colors = {'red': 31, 'green': 32, 'yellow': 33, 'blue': 34, 'magenta': 35, 'cyan': 36, 'white': 37, 'bright_red': 91, 
                   'bright_green': 92, 'bright_yellow': 93, 'bright_blue': 94, 'bright_magenta': 95, 'bright_cyan': 96, 
                   'bright_white': 97, 'black': 30, 'gray': 90,}

    bg_colors = {'red': 41, 'green': 42, 'yellow': 43, 'blue': 44, 'magenta': 45, 'cyan': 46, 'white': 47, 'bg_bright_red': 101, 
                 'bg_bright_green': 102, 'bg_bright_yellow': 103, 'bg_bright_blue': 104, 'bg_bright_magenta': 105, 
                 'bg_bright_cyan': 106, 'bg_bright_white': 107, 'bg_black': 40, 'bg_gray': 100,}
    
    if not isinstance(content, str):
        try:
            content = str(content)
        except:
            xprint("⚠️  The content doesn't convert into string, some functions don't work!", color='red')
        
        if not isinstance(content, str):
            # 清空终端内容
            if clear:
                screen_clear(clear=clear)
            
            # 直接打印
            if not pprint:
                print(content)
            else:
                _pprint.pprint(content)
                
            return
        
    start_code = ''  # 开始的转义码
    end_code = '\033[0m'  # 结束的转义码
    
    # 设置字体颜色
    if color:
        color = color.lower()
        if color.lower() == 'random':
            start_code += f'\033[{random.randint(31, 97)}m'
        else:
            start_code += f'\033[{font_colors[color]}m'

    # 设置背景颜色
    if bg_color:
        bg_color = bg_color.lower()
        if bg_color.lower() == 'random':
            start_code += f'\033[{random.randint(41, 107)}m'
        else:
            start_code += f'\033[{bg_colors[bg_color]}m'
        
    # 设置下划线
    if underline:
        start_code += '\033[4m'
        
    # 设置加粗
    if bold:
        start_code += '\033[1m'

    # 清空终端内容
    if clear:
        screen_clear(clear=clear)
        
    # 如果需要添加水平线
    if hl:
        if hl_style == 'full':  # 打印终端宽度的水平线
            terminal_width = shutil.get_terminal_size((80, 20)).columns  # 获取终端宽度
            hl = hl * terminal_width  # 根据终端宽度打印水平线
            # 打印水平线
            xprint(hl, color=color, bg_color=None, underline=False, bold=False, end='\n', 
                   hl=False)
            
        if hl_style == 'paragraph':  # 根据内容打印合适宽度的水平线
            # 根据换行符分割
            lines = content.split("\n")
            max_len_line = max(lines, key=find_text_place_length)
            line_len = find_text_place_length(max_len_line)
            hl = hl * line_len
            # 打印水平线
            xprint(hl, color=color, bg_color=None, underline=False, bold=False, end='\n', 
                   hl=False)

    # 打印内容
    print(start_code + content + end_code, end=end)
    
    if hl and hl_num > 1:  # 添加另外的水平线
        xprint(hl, color=color, bg_color=None, underline=False, bold=False, end='\n', 
                hl=False)
        

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
    parma1 = "images"
    param2 = "output_images"
    param3 = 2
    param4 = dict(
        p1='abc',
        p2=123
    )
    param5 = ['1', 'abc']
    
    print_arguments(parma1, param2, param3, wait=True, table_verbose=False, param_type=True)

    # 使用示例
    # xprint("这是一段普通的文本")
    # xprint("这是一段红色文本", color='red')
    # xprint("这是一段带下划线的文本", underline=True)
    # xprint("这是一段加粗的文本", bold=True)
    # xprint("这是一段加粗的文本", bold=True)
    # xprint("这是一段黄色加粗带下划线的文本", color='yellow', underline=True, bold=True)
    # xprint("这是一段黄色加粗带下划线的文本\n", color='yellow', underline=True, bold=True, horizontal_line="-")
    xprint("This is a line\这是第二行，会比第一行长很多 (more and more)！！！！！", 
           color='yellow', underline=True, bold=True, hl=">", hl_num=2)
    xprint("This is a test", color='random', bg_color='random', underline=True, bold=True)
    xprint(param4, color='random', bg_color='random', underline=True, bold=True, hl="<")
    xprint(param5, color='random', bg_color='random', underline=True, bold=True, hl="<")


    # 使用默认格式获取当前时间
    print(get_current_time())
    print(get_current_time('format1'))
    print(get_current_time('format2'))
    print(get_current_time('format3'))
    print(get_current_time('format3'))
    print(get_current_time('%Y%m%d-%H%M'))