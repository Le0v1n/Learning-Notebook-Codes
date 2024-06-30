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
    parser = argparse.ArgumentParser(
        prog='ProgramName',  # 指定程序名称为 'ProgramName'
        description='What the program does',  # 程序的描述
        epilog='Text at the bottom of help'  # 帮助信息底部的文本
    )

    # 为解析器对象添加参数
    parser.add_argument('config', help='The path of config.')  # 位置参数
    parser.add_argument('--dataset', type=str, default='Datasets/coco128', help='The dir path of dataset.') 
    parser.add_argument('--weights-path', type=str, default='runs/detect/exp/weights/best.pt', help='The path of model weights.') 
    parser.add_argument('-e', '--epoch', type=int, default=150, help='The epoch of training.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='The learning rate of optimizer.')
    parser.add_argument('-c', '--count')  # 
    parser.add_argument('-v', '--verbose', action='store_true')  # on/off的一个flag

    # 让解析器对象解析参数
    args = parser.parse_args()
    
    return args, parser


if __name__ == '__main__':
    args, parser = get_opts()
    print_args(args, parser)
    
