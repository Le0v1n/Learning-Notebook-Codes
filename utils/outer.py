import os
import sys
import shutil
import prettytable
import random
import pprint as _pprint
import argparse
import re
import math
from typing import Any

sys.path.append(os.getcwd())


def __screen_clear():
    # 检查操作系统类型
    if os.name == 'nt':
        # Windows系统
        os.system('cls')
    elif os.name == 'posix':
        # Linux系统
        os.system('clear')
    else:
        # 其他的操作系统
        print("不清除屏幕，当前操作系统不受支持。")


def find_text_place_length(text):
    """找到一段文字中的长度：英文1位，中文2位

    Args:
        text (str): 文字

    Returns:
        int: 长度
    """
    pattern_alpha = re.compile(r'[a-zA-Z]')  # 匹配英文字母的正则表达式
    pattern_digit = re.compile(r'\d')  # 匹配数字的正则表达式
    pattern_punct = re.compile(r'[.,;:!?()"]')  # 匹配英文标点符号的正则表达式

    alpha_matches = len(pattern_alpha.findall(text))  # 找到所有英文字母的匹配项数量
    digit_matches = len(pattern_digit.findall(text))  # 找到所有数字的匹配项数量
    punct_matches = len(pattern_punct.findall(text))  # 找到所有英文标点符号的匹配项数量

    return 2 * len(text) - alpha_matches - digit_matches - punct_matches


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
                __screen_clear(clear=clear)

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
        __screen_clear(clear=clear)

    # 如果需要添加水平线
    if hl:
        terminal_width = shutil.get_terminal_size((80, 20)).columns  # 获取终端宽度
        if hl_style == 'full':  # 打印终端宽度的水平线
            hl = hl * terminal_width  # 根据终端宽度打印水平线
            # 打印水平线
            xprint(hl, color=color, bg_color=None, underline=False, bold=False, end='\n',
                   hl=False)

        if hl_style == 'paragraph':  # 根据内容打印合适宽度的水平线
            # 根据换行符分割
            lines = content.split("\n")
            max_len_line = max(lines, key=find_text_place_length)
            line_len = max(5, find_text_place_length(max_len_line))
            line_len = min(line_len, terminal_width)
            hl = hl * line_len
            # 打印水平线
            xprint(hl, color=color, bg_color=None, underline=False, bold=False, end='\n',
                   hl=False)

    # 打印内容
    if pprint:
        _pprint.pprint(content)
    else:
        print(start_code + content + end_code, end=end)

    if hl and hl_num > 1:  # 添加另外的水平线
        xprint(hl, color=color, bg_color=None, underline=False, bold=False, end='\n',
                hl=False)
        
        
def __find_max_length(*args, **kwargs):
    kwargs = kwargs['kwargs']
    
    lengths_name = []
    lengths_type = []
    
    for value in args:
        length_type = find_text_place_length(str(type(value)))
        lengths_type.append(length_type)
        
    for name, value in kwargs.items():
        length_name = find_text_place_length(str(name))
        lengths_name.append(length_name)
        
        length_type = find_text_place_length(str(type(value)))
        lengths_type.append(length_type)
        
    G_max_length_name = max(max(lengths_name), find_text_place_length('name')) if lengths_name else 0
    G_max_length_type = max(max(lengths_type), find_text_place_length('type')) if lengths_type else 0
    
    return G_max_length_name, G_max_length_type


def __add_row_to_table(table: prettytable.prettytable.PrettyTable, 
                     index: int, name: str, value: Any, 
                     max_length_type: int, max_length_name: int, 
                     terminal_width: int = 80, 
                     offset: int = 4, show_type: bool = False):
    if not name:
        name = ""
    
    __index = index
    __type = type(value)
    __name = name
    
    # 获取除 value 外的长度
    len_index = max(find_text_place_length('index'), find_text_place_length(str(__index))) + offset
    len_type = max(max_length_type, find_text_place_length(str(__type))) + offset if show_type else 0
    len_name = max(max_length_name, find_text_place_length(str(__name))) + offset
    
    len_total_except_value = len_index + len_type + len_name
    
    # 剩余可以给 value 的长度
    len_rest = terminal_width - len_total_except_value
    
    # 求出 value 需要的长度
    len_value_need = find_text_place_length(str(value)) + offset
    
    # 求出换行次数
    enter_times = math.floor(len_value_need / len_rest)
    
    # 对 value 进行拆分
    value_str = str(value)
    
    # 得到每行所需的索引
    lines = []
    start = 0
    for _ in range(enter_times):
        # 计算每行的结束索引
        end = start + len_rest - offset  # 减去末尾的空格
        
        # 如果结束索引超出了字符串长度，则设置为字符串长度
        end = min(end, len(value_str))
        
        # 添加到行列表
        lines.append(value_str[start:end])
        
        # 更新下一行的开始索引
        start = end
        
    # 如果有剩余的字符，添加到最后一个段落
    if start < len(value_str):
        lines.append(value_str[start:])
        
    # 输出结果
    for idx, line in enumerate(lines):
        if idx == 0:
            table.add_row([__index, __type, __name, line])
        else:
            table.add_row(["", "", "", line])


def print_arguments(*args, **kwargs) -> prettytable.prettytable.PrettyTable:
    """打印参数的函数

    - Args:
        - *args: 直接传参 --> print_arguments(param1, param2, ...)
        - **kwargs: 有关键字的传参 --> print_arguments(key1=param1, key2=param2)
    
    - Return
        - 根据 `return_object (bool) = False` 决定返回的是 `prettytable.prettytable.PrettyTable` 对象还是字符串

    - 💡  Tips: 
        1. `argparse=`: 自动处理 `argparse.Namespace` 对象
        2. `params_dict=[[param1, param2], ...]` 则会自动打印
        3. `wait=True` 或 `confirm=True` 或 `check_True` 或 `check_params`，则会等待用户输入 `yes` 后程序会继续执行，目的是检查参数是否正确
        4. `show_type=True`: 则会显示传参时的参数类型
        5. `only_confirm/only_wait=True`: 仅开启参数检查
        6. `silent=True`: 不打印
        7. `return_object=True`: 返回一个 `prettytable.prettytable.PrettyTable` 对象，否则返回 PrettyTable.tostring() 字符串
        8. 如果某一行参数过长，则该行会被拆分为多行（不影响 `index` 的顺序）
    """
    # ========== 根据字典内容创建一些 flag，并在字典中删除这些键值对 ==========
    flag_only_wait = any(kwargs.get(key, False) for key in ('only_wait', 'only_confirm', 'only_check', 'only_check_params'))
    flag_wait = any(kwargs.get(key, False) for key in ('wait', 'confirm', 'check', 'check_param', 'check_params'))
    flag_show_type = kwargs.get('show_type', False)
    flag_silent = kwargs.get('silent', False)
    flag_return_object = kwargs.get('return_object', False)

    # 删除上面涉及到的键值对
    for key in ('wait', 'confirm', 'check', 'check_param', 'check_params', 'show_type', 'silent', 'return_object',
                'only_wait', 'only_confirm', 'only_check', 'only_check_params'):
        if key in kwargs:
            del kwargs[key]
    
    # ========== 处理 only_confirm | only_wait ==========
    if flag_only_wait:
        user_input = input("\033[1;31mContinue (Yes/Y)?  \033[0m").lower()
        if user_input in ['yes', 'y']:  # 继续
            return
        elif user_input == 'no' or user_input == 'n':  # 退出
            sys.exit("User exit!\n")
        else:  # 退出
            print("Invalid input!")
            sys.exit("User exit!")

    # ========== 创建 PrettyTable 对象 ==========
    table = prettytable.PrettyTable(["index", "type", "name", "value"])
    table.border = kwargs.get("table_border", True)
    table.align["index"] = 'c'
    table.align["type"] = 'c'
    table.align["name"] = 'l'
    table.align["value"] = 'l'

    # ========== 获取终端长度 ==========
    terminal_width = shutil.get_terminal_size((80, 20)).columns  # 获取终端宽度
    
    # ========== 全局变量  ==========
    G_index = 1  # PrettyTable 的索引

    # 找到全局的最大长度: name、type
    G_max_length_name, G_max_length_type = __find_max_length(args, kwargs=kwargs)
    
    # ========== 处理 *args 参数 ==========
    for arg in args:
        __add_row_to_table(table, G_index, None, arg, G_max_length_type, G_max_length_name, terminal_width,
                         offset=7, show_type=flag_show_type)
        G_index += 1

    # ========== 处理 argparse.Namespace ==========
    _args = kwargs.get('argparse', None)
    if isinstance(_args, argparse.Namespace):
        for k, v in vars(_args).items():
            __add_row_to_table(table, G_index, k, v, G_max_length_type, G_max_length_name, terminal_width,
                            offset=7, show_type=flag_show_type)
            G_index += 1
        del kwargs['argparse']

    # ========== 处理 params_dict=[[param1, param2], ...] ==========
    params_dict = kwargs.get('params_dict', None)
    if isinstance(params_dict, list):  # 判断是不是list
        for value in params_dict:
            __add_row_to_table(table, G_index, None, value, G_max_length_type, G_max_length_name, terminal_width,
                            offset=7, show_type=flag_show_type)
            G_index += 1
        del kwargs['params_dict']
    
    # ========== 处理 **kwargs 参数 ==========
    for key, value in kwargs.items():
        __add_row_to_table(table, G_index, key, value, G_max_length_type, G_max_length_name, terminal_width,
                         offset=7, show_type=flag_show_type)
        G_index += 1

    # ========== 处理 show_type 参数 ==========
    if not flag_show_type:
        table.del_column(fieldname='type')

    # ========== 处理 silent=True ==========
    if not flag_silent and len(table.rows) > 0:
        print(table)
        
    # ========== 处理 wait=True, confirm=True ==========
    if flag_wait:
        print_arguments(only_wait=True)
    
    # ========== 处理 return_object=True ==========
    if flag_return_object:
        return table
    else:
        return table.get_string()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='这是一个示例程序。')
    
    # 添加参数
    parser.add_argument('--integer', type=float, default=3.1415926, help='一个整数参数')
    parser.add_argument('--file', type=str, default='Example....', help='一个文件路径参数')
    parser.add_argument('-d', '--debug', action='store_true', help='开启调试模式')
    
    # 解析参数
    args = parser.parse_args()
    
    print_arguments(
        123,
        ['person', 'tvmonitor', 'chair', 'bottle', 'dog', 'horse', 'train', 'motorbike', 'bicycle', 'pottedplant', 'car', 'bird', 'sheep', 'bus', 'sofa', 'cat', 'boat', 'diningtable', 'cow', 'aeroplane'],
        'abc','1231232133333333333331231232133333333333331231232133333333333331231232'
        '13333333333333123123213333333333333123123213333333333333123123213333333333333'
        '123123213333333333333123123213333333333333123123213333333333333123123213333333'
        '333333123123213333333333333123123213333333333333123123213333333333333123123213333333333333',
        param='exaple',
        example_param1 = ['person', 'tvmonitor', 'chair', 'bottle', 'dog', 'horse', 'train', 'motorbike', 'bicycle', 'pottedplant', 'car', 'bird', 'sheep', 'bus', 'sofa', 'cat', 'boat', 'diningtable', 'cow', 'aeroplane'],
        example_param = ['person', 'bicycle', 'pottedplant', 'car', 'bird', 'sheep', 'bus', 'sofa', 'cat', 'boat', 'diningtable', 'cow', 'aeroplane'],
        params_dict = [123, 'aaa'],
        argparse=args,
        show_type=True
    )