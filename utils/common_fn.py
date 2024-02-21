import os
import sys
import shutil
import prettytable
import re
import random
import pprint as _pprint


def print_arguments(*args, **kwargs):
    """打印参数的函数
    
    *args: 直接传参 --> print_arguments(param1, param2, ...)
    **kwargs: 有关键字的传参 --> print_arguments(key1=param1, key2=param2)
    
    💡Node: 
        1. 如果传入 wait=True，则会等待用户输入yes后程序会继续执行，目的是检查参数是否正确
        2. table_verbose: 则会显示传参时的参数类型
    """
    table = prettytable.PrettyTable(["param type", "index", "name", "type", "value"])
    table.border = kwargs.get("table_border", True)
    table.align["param type"] = 'l'
    table.align["index"] = 'c'
    table.align["name"] = 'l'
    table.align["type"] = 'c'
    table.align["value"] = 'l'
    
    
    # 添加*args参数
    for arg in args:
        table.add_row(['*args', f"{len(table.rows)+1}", "", type(arg), arg])
    
    # 添加**kwargs
    for k, v in kwargs.items():
        table.add_row(['**kwargs', len(table.rows)+1, k, type(v), v])
    
    if not kwargs.get("table_verbose", False):
        table.del_column(fieldname='param type')
        
    if not kwargs.get("param_type", False):
        table.del_column(fieldname='type')
        
    print(table)
    
    if kwargs.get('wait', False):
        while True:
            user_input = input("\033[1;31mContinue (Yes/Y)?  \033[0m").lower()
            if user_input in ['', 'yes', 'y']:
                break
            elif user_input == 'no' or user_input == 'n':
                sys.exit("User exit!\n")
            else:
                print("Invalid input!\n")
                
                
def create_folder(fp, exist_ok=True):
    """创建文件夹
        当父级文件夹存在是，调用os.mkdir方法
        当父级文件夹不存在时，调用os.makedirs方法

    Args:
        fp (str): 要创建的文件夹路径
        exist_ok (bool, optional): 建议为True，如果为False，当fp存在时可能会报错. Defaults to True.
    """
    if os.path.exists(fp):
        return
        
    # get parent folder path
    parent_fp = os.path.dirname(fp)
    
    if not os.path.exists(parent_fp):
        print(f"⚠️ The parent folder doesn't exists!")
        os.makedirs(fp, exist_ok=exist_ok)
    else:
        os.mkdir(fp)
        
        
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


def get_file_size(fp, ndigits=4):
    """获取文件大小

    Args:
        file_path (str): 图片路径

    Returns:
        float: 文件大小（MB）
    """
    # 获取文件大小（字节）
    file_size_bytes = os.path.getsize(fp)

    # 将文件大小转换为 MB
    file_size_mb = round(file_size_bytes / (1024 * 1024), ndigits=ndigits)

    return file_size_mb


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
           horizontal_line='', horizontal_line_length='paragraph', clear=False, pprint=False):
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
        # 清空终端内容
        if clear:
            screen_clear(clear=clear)
        
        # 直接打印
        if not pprint:
            print(content)
        else:
            _pprint.pprint(content)
            
        xprint("⚠️ The content doesn't string, some functions don't work!", color='red')
        return
        
    start_code = ''  # 开始的转义码
    end_code = '\033[0m'  # 结束的转义码
    
    # 设置字体颜色
    if color:
        if color.lower() == 'random':
            start_code += f'\033[{random.randint(31, 97)}m'
        else:
            start_code += f'\033[{font_colors[color]}m'

    # 设置背景颜色
    if bg_color:
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

    # 如果需要添加水平线
    if horizontal_line:
        if horizontal_line_length == 'full':  # 打印终端宽度的水平线
            terminal_width = shutil.get_terminal_size((80, 20)).columns  # 获取终端宽度
            start_code = horizontal_line * terminal_width + '\n' + start_code  # 根据终端宽度打印水平线
        if horizontal_line_length == 'paragraph':  # 根据内容打印合适宽度的水平线
            # 根据换行符分割
            lines = content.split("\n")
            max_len_line = max(lines, key=find_text_place_length)
            line_len = find_text_place_length(max_len_line)
            start_code = horizontal_line * line_len + '\n' + start_code

    # 清空终端内容
    if clear:
        screen_clear(clear=clear)

    # 打印内容
    print(start_code + content + end_code, end=end)
    

if __name__ == "__main__":
    parma1 = "images"
    param2 = "output_images"
    param3 = 2
    
    print_arguments(parma1, param2, param3, wait=True, table_verbose=False, param_type=True)

    # 使用示例
    # xprint("这是一段普通的文本")
    # xprint("这是一段红色文本", color='red')
    # xprint("这是一段带下划线的文本", underline=True)
    # xprint("这是一段加粗的文本", bold=True)
    # xprint("这是一段加粗的文本", bold=True)
    # xprint("这是一段黄色加粗带下划线的文本", color='yellow', underline=True, bold=True)
    # xprint("这是一段黄色加粗带下划线的文本\n", color='yellow', underline=True, bold=True, horizontal_line="-")
    xprint("This is a line\这是第二行，会比第一行长很多 (more and more)！！！！！", color='yellow', underline=True, bold=True, horizontal_line=">")
    xprint("This is a test", color='random', bg_color='random', underline=True, bold=True)
