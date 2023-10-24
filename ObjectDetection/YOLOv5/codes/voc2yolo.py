"""
本脚本有两个功能：
    1. 将 voc 数据集标注信息(.xml)转为 yolo 标注格式(.txt)，并将图像文件复制到相应文件夹
    2. 根据 json 标签文件，生成对应 names 标签(my_data_label.names)
    3. 兼容 YOLOv3 和 YOLOv5
"""
import os
from tqdm import tqdm
from lxml import etree
import json
import shutil
import argparse
from tqdm import tqdm
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split


def args_table(args):
    # 创建一个表格
    table = PrettyTable(["Parameter", "Value"])
    table.align["Parameter"] = "l"  # 使用 "l" 表示左对齐
    table.align["Value"] = "l"  # 使用 "l" 表示左对齐

    # 将args对象的键值对添加到表格中
    for key, value in vars(args).items():
        # 处理列表的特殊格式
        if isinstance(value, list):
            value = ', '.join(map(str, value))
        table.add_row([key, value])

    # 返回表格的字符串表示
    return str(table)


def generate_train_and_val_txt(args):
    
    target_train_file = args.train_txt_path
    target_val_file = args.val_txt_path

    # 获取源文件夹中的所有文件
    files = os.listdir(args.voc_images_path)
    
    # 划分训练集和验证集
    train_images, val_images = train_test_split(files, test_size=args.val_size, random_state=args.seed)

    # 打开目标文件以写入模式
    with open(target_train_file, 'w', encoding='utf-8') as f:
        # 使用tqdm创建一个进度条，迭代源文件列表
        for file in tqdm(train_images, desc=f"\033[1;33mProcessing Files for train\033[0m"):
            file_name, _ = os.path.splitext(file)
            # 写入文件名
            f.write(f'{file_name}\n')

    with open(target_val_file, 'w', encoding='utf-8') as f:
        # 使用tqdm创建一个进度条，迭代源文件列表
        for file in tqdm(val_images, desc=f"\033[1;33mProcessing Files for val\033[0m"):
            file_name, _ = os.path.splitext(file)
            # 写入文件名
            f.write(f'{file_name}\n')

    print(f"\033[1;32m文件名已写入到 {target_train_file} 和 {target_val_file} 文件中!\033[0m")

def parse_args():
    # 创建解析器
    parser = argparse.ArgumentParser(description="将 .xml 转换为 .txt")
    
    # 添加参数
    parser.add_argument('--voc_root', type=str, default="VOCdevkit", help="PASCAL VOC路径(之后的所有路径都在voc_root下)")
    parser.add_argument('--voc_version', type=str, default="VOC2012-Lite", help="VOC 版本")
    parser.add_argument('--save_path', type=str, default="VOC2012-YOLO", help="转换后的保存目录路径")
    parser.add_argument('--train_list_name', type=str, default="train.txt", help="训练图片列表名称")
    parser.add_argument('--val_list_name', type=str, default="val.txt", help="验证图片列表名称")
    parser.add_argument('--val_size', type=float, default=0.1, help="验证集比例")
    parser.add_argument('--seed', type=int, default=42, help="随机数种子")
    parser.add_argument('--no_create_txt_for_pure_negative_sample', action='store_true', help='是否为纯负样本创建txt文件(默认创建)')
    parser.add_argument('--num_classes', type=int, default=20, help="数据集类别数(用于校验)")
    parser.add_argument('--classes', help="数据集具体类别数(用于生成 classes.json 文件)", 
                        default=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                                 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
                                 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])
    
    return parser.parse_args()


def configure_path(args):
    # 转换的训练集以及验证集对应txt文件
    args.train_txt = "train.txt"
    args.val_txt = "val.txt"

    # 转换后的文件保存目录
    args.save_file_root = os.path.join(args.voc_root, args.save_path)

    # 生成json文件
    # label标签对应json文件
    args.label_json_path = os.path.join(args.voc_root, "classes.json")
    
    # 创建一个将类别与数值关联的字典
    class_mapping = {class_name: index + 1 for index, class_name in enumerate(args.classes)}
    with open(args.label_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(class_mapping, json_file, ensure_ascii=False, indent=4)

    print(f'\033[1;31m类别列表已保存到 {args.label_json_path}\033[0m')

    # 拼接出voc的images目录，xml目录，txt目录
    args.voc_images_path = os.path.join(args.voc_root, args.voc_version, "JPEGImages")
    args.voc_xml_path = os.path.join(args.voc_root, args.voc_version, "Annotations")
    args.train_txt_path = os.path.join(args.voc_root, args.voc_version, args.train_txt)
    args.val_txt_path = os.path.join(args.voc_root, args.voc_version, args.val_txt)
    
    # 生成对应的 train.txt 和 val.txt
    generate_train_and_val_txt(args)

    # 检查文件/文件夹都是否存在
    assert os.path.exists(args.voc_images_path), f"VOC images path not exist...({args.voc_images_path})"
    assert os.path.exists(args.voc_xml_path), f"VOC xml path not exist...({args.voc_xml_path})"
    assert os.path.exists(args.train_txt_path), f"VOC train txt file not exist...({args.train_txt_path})"
    assert os.path.exists(args.val_txt_path), f"VOC val txt file not exist...({args.val_txt_path})"
    assert os.path.exists(args.label_json_path), f"label_json_path does not exist...({args.label_json_path})"
    if os.path.exists(args.save_file_root) is False:
        os.makedirs(args.save_file_root)
        print(f"创建文件夹：{args.save_file_root}")


def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args：
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def translate_info(file_names: list, save_root: str, class_dict: dict, train_val='train', args=None):
    """
    将对应xml文件信息转为yolo中使用的txt文件信息
    """
    save_txt_path = os.path.join(save_root, train_val, "labels")
    if os.path.exists(save_txt_path) is False:
        os.makedirs(save_txt_path)
    save_images_path = os.path.join(save_root, train_val, "images")
    if os.path.exists(save_images_path) is False:
        os.makedirs(save_images_path)

    for file in tqdm(file_names, desc="translate {} file...".format(train_val)):
        # 检查下图像文件是否存在（强报错！）
        img_path = os.path.join(args.voc_images_path, file + ".jpg")
        assert os.path.exists(img_path), "file:{} not exist...".format(img_path)

        # 检查xml文件是否存在（强报错！）
        xml_path = os.path.join(args.voc_xml_path, file + ".xml")
        assert os.path.exists(xml_path), "file:{} not exist...".format(xml_path)

        # 读取 xml 文件（这里修复了一下）
        # with open(xml_path) as fid:
        #     xml_str = fid.read()
        # xml = etree.fromstring(xml_str)

        with open(xml_path) as fid:
            xml_str = fid.read()
            
        # 将XML字符串编码为字节序列
        xml_bytes = xml_str.encode('utf-8')

        # 使用lxml解析字节序列的XML数据
        xml = etree.fromstring(xml_bytes)
        data = parse_xml_to_dict(xml)["annotation"]
        img_height = int(data["size"]["height"])
        img_width = int(data["size"]["width"])

        # write object info into txt
        # assert "object" in data.keys(), "file: '{}' lack of object key.".format(xml_path)
        if (not "object" in data.keys()) or (len(data["object"]) == 0):  # 没有目标，说明是纯负样本
            if args.no_create_txt_for_pure_negative_sample:  # 不为纯负样本创建txt文件
                continue
            else:  # 为纯负样本创建txt文件
                # 把纯负样本图片拷贝到指定为止
                path_copy_to = os.path.join(save_images_path, img_path.split(os.sep)[-1])
                if os.path.exists(path_copy_to) is False:
                    shutil.copyfile(img_path, path_copy_to)
                
                # 创建一个空的 .txt 文件
                with open(os.path.join(save_txt_path, file + ".txt"), "w") as f:
                    ...

                # 后面的不需要执行
                continue
                
        with open(os.path.join(save_txt_path, file + ".txt"), "w") as f:
            for index, obj in enumerate(data["object"]):
                # 获取每个object的box信息
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                class_name = obj["name"]
                
                class_index = class_dict[class_name] - 1  # 目标id从0开始

                # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
                if xmax <= xmin or ymax <= ymin:
                    print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                    continue

                # 将box信息转换到yolo格式
                xcenter = xmin + (xmax - xmin) / 2
                ycenter = ymin + (ymax - ymin) / 2
                w = xmax - xmin
                h = ymax - ymin

                # 绝对坐标转相对坐标，保存6位小数
                xcenter = round(xcenter / img_width, 6)
                ycenter = round(ycenter / img_height, 6)
                w = round(w / img_width, 6)
                h = round(h / img_height, 6)

                info = [str(i) for i in [class_index, xcenter, ycenter, w, h]]

                if index == 0:
                    f.write(" ".join(info))
                else:
                    f.write("\n" + " ".join(info))

        # copy image into save_images_path
        path_copy_to = os.path.join(save_images_path, img_path.split(os.sep)[-1])
        if os.path.exists(path_copy_to) is False:
            shutil.copyfile(img_path, path_copy_to)


def create_class_names(class_dict: dict, args):
    keys = class_dict.keys()
    with open(os.path.join(args.voc_root, "my_data_label.names"), "w") as w:
        for index, k in enumerate(keys):
            if index + 1 == len(keys):
                w.write(k)
            else:
                w.write(k + "\n")


def main(args):
    # read class_indict
    json_file = open(args.label_json_path, 'r')
    class_dict = json.load(json_file)

    # 读取train.txt中的所有行信息，删除空行
    with open(args.train_txt_path, "r") as r:
        train_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
        
    # voc信息转yolo，并将图像文件复制到相应文件夹
    translate_info(train_file_names, args.save_file_root, class_dict, "train", args=args)

    # 读取val.txt中的所有行信息，删除空行
    with open(args.val_txt_path, "r") as r:
        val_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
        
    # voc信息转yolo，并将图像文件复制到相应文件夹
    translate_info(val_file_names, args.save_file_root, class_dict, "val", args=args)

    # 创建my_data_label.names文件
    create_class_names(class_dict, args=args)


if __name__ == "__main__":
    args = parse_args()
    configure_path(args)
    
    # 美化打印 args
    print(f"\033[1;34m{args_table(args)}\033[0m")
    
    # 执行 .xml 转 .txt
    main(args)