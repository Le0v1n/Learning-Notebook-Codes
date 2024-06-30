import argparse
from prettytable import PrettyTable
import re


def get_help_for_argument(parser, arg_name):
    for action in parser._actions:
        if action.dest == arg_name or (action.option_strings and any(opt == '--'+arg_name for opt in action.option_strings)):
            return action.help
    return "N/A"


def print_args(args, parser):
    ptable = PrettyTable(['Argument', 'Type', 'Value', 'Help/Description'])
    ptable.align = 'l'
    

    for name, value in args._get_kwargs():
        ptable.add_row(
            [
                name, 
                re.findall(pattern=r"<class '(.*)'>", string=str(type(value)))[0], 
                value, 
                get_help_for_argument(parser=parser, arg_name=name)
            ]
        )
    
    print(ptable)


def get_opts():
    # 创建一个解析器对象
    parser = argparse.ArgumentParser(description='The example for Le0v1n article')

    # 为解析器对象添加参数
    parser.add_argument('--mode', action='store_const', const='DEBUG', default='INFO', help='Set the mode to const value')

    # 让解析器对象解析参数
    args = parser.parse_args()
    
    return args, parser


if __name__ == '__main__':
    args, parser = get_opts()
    print_args(args, parser)
    
