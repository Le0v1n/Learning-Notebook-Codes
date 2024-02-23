"""
+ 脚本说明：根据图片修改xml文件中的size尺寸信息
+ 用途：修正数据集标签的<size>信息
+ 要求：无
+ 注意：
  1. 不是in-place操作
  2. 不需要转换的也会复制到新的文件夹下
  3. 如果遇到xml没有对应图片的，则会记录该错误，并生成 ERROR_LOG.txt 文件
"""
from PIL import Image
import os
import xml.etree.ElementTree as ET
import tqdm
import sys


"""============================ 需要修改的地方 ==================================="""
# 输入和输出文件夹路径
XML_PATH  = "EXAMPLE_FOLDER/labels-xml"  # 修正前的 xml 文件夹路径
SAVE_PATH = "EXAMPLE_FOLDER/labels-xml-fixed"  # 修正后的 xml 文件夹路径
IMG_PATH  = "EXAMPLE_FOLDER/images"  # 同名图片文件夹路径
img_type  = '.jpg'  # 图片的格式
"""==============================================================================="""

# 确保输出文件夹存在
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH, exist_ok=True)

# 获取xml文件列表
annotation_files = [file for file in os.listdir(XML_PATH) if file.lower().endswith('.xml')]

"------------计数------------"
TOTAL_NUM   = len(annotation_files)  # 需要处理的 .xml 文件数量
SUCCEED_NUM = 0  # 成功修改的数量
SKIP_NUM    = 0  # 跳过的数量
ERROR_NUM   = 0  # 出错的数量
ERROR_LIST  = []  # 出错的logging
"---------------------------"

# 遍历所有的xml文件
process_bar = tqdm.tqdm(total=TOTAL_NUM, desc="根据图片修正 xml 文件的尺寸 <size> 信息", unit='xml')
for xml_file in annotation_files:
    xml_name, xml_ext = os.path.splitext(xml_file)  # 分离文件名和后缀
    process_bar.set_description(f"Process in \033[1;31m{xml_file}\033[0m")
    
    # 读取 xml 文件
    xml_path = os.path.join(XML_PATH, xml_file)  # 获取完整路径
    tree     = ET.parse(xml_path)  # 解析 xml 树
    root     = tree.getroot()  # 获取 xml 树的根
    
    # 获取同名图片文件名
    image_path = os.path.join(IMG_PATH, xml_name) + img_type
    
    # 判断对应的同名图片文件是否存在，如果不存在则记录错误
    if not os.path.exists(image_path):
        ERROR_NUM += 1
        ERROR_LIST.append(xml_path)
        process_bar.update()
        continue
    
    # 使用PIL获取图片尺寸
    image = Image.open(image_path)
    width, height = image.size
    
    # 判断 xml 中的 <size> 标签是否和图片尺寸对应
    size_elem = root.find("size")
    if size_elem.find("width").text == str(width) and size_elem.find("height").text == str(height):
        # 不需要修正，直接保存文件
        output_path = os.path.join(SAVE_PATH, xml_file)
        tree.write(output_path, encoding="utf-8")
        SKIP_NUM += 1
        process_bar.update()
        continue
    else:
        # 更新xml中的<size>标签
        size_elem.find("width").text = str(width)
        size_elem.find("height").text = str(height)

        # 保存修正后的xml文件
        output_path = os.path.join(SAVE_PATH, xml_file)
        tree.write(output_path, encoding="utf-8")
        SUCCEED_NUM += 1
        process_bar.update()
process_bar.close()

print(f"👌 xml 文件的 size 信息修正已完成, 详情如下:"
      f"\n\t成功修正数量/总xml数量 = \033[1;32m{SUCCEED_NUM}\033[0m/{TOTAL_NUM}"
      f"\n\t跳过数量/总xml数量 = \033[1;34m{SKIP_NUM}\033[0m/{TOTAL_NUM}"
      f"\n\t出错数量/总xml数量 = \033[1;31m{ERROR_NUM}\033[0m/{TOTAL_NUM}")

if SUCCEED_NUM + SKIP_NUM == TOTAL_NUM:
    print("👌 \033[1;32mNo Problems\033[0m")
else:
    print(f"🤡 貌似有点问题, 请仔细核查!"
          f"\n\tSUCCEED_NUM: {SUCCEED_NUM}"
          f"\n\tSKIP_NUM: {SKIP_NUM}"
          f"\n\tERROR_NUM = {ERROR_NUM}"
          f"\nSUCCEED_NUM + SKIP_NUM + ERROR_NUM = {SUCCEED_NUM + SKIP_NUM + ERROR_NUM}"
          f"\nTOTAL_NUM: {TOTAL_NUM}")

if ERROR_LIST:  # 如果有出错信息
    program_path = sys.argv[0]  # 获取程序完整路径
    program_name = os.path.basename(program_path)  # 获取程序名称
    program_parent_path = os.path.dirname(program_path)  # 获取程序所在文件夹路径
    
    ERROR_LOG_PATH = os.path.join(program_parent_path, f"ERROR_LOG-[{program_name}].txt")
    
    with open(ERROR_LOG_PATH, "w") as file:  # 打开文本文件以写入模式
        file.write(f"Program: {program_path}\n")  # 写入程序名称
        file.write(f"🤡 出错了 -> 出错数量/总文件数量 = {ERROR_NUM}/{TOTAL_NUM}\n")  # 写入总体出错信息
        file.write('=' * 50 + '\n')  # 写入分隔线

        # 遍历出错信息列表，写入文件
        for e in ERROR_LIST:
            file.write(f"{e}\n")
            
        # 写入分隔线
        file.write('=' * 50 + '\n')
        
    print(f"\033[1;31m出错信息\033[0m已写入到 [\033[1;34m{ERROR_LOG_PATH}\033[0m] 文件中, 请注意查看!")