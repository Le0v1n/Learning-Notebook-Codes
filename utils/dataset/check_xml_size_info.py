import os
import sys
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm.rich import tqdm

sys.path.append(os.getcwd())
from utils.common_fn import xprint, print_arguments, get_logger, get_logger_save_path
from utils.file_type import ImageFormat


__doc__ = """脚本说明：根据图片修改xml文件中的size尺寸信息
    用途：修正数据集标签的<size>信息
    要求：无
    注意：
        1. 不是in-place操作
        2. 不需要转换的也会复制到新的文件夹下
        3. 如果遇到xml没有对应图片的，则会记录该错误，并生成 ERROR_LOG.txt 文件
"""


def check_xml_size_info(src_xmls_path: str, 
                        dst_xmls_save_path: str,
                        src_images_path: str, 
                        src_images_type: tuple = ImageFormat,
                        verbose=False,
                        confirm: bool = True):
    """根据图片修改xml文件中的size尺寸信息

    Args:
        src_xmls_path (str): xmls的文件夹路径
        dst_xmls_save_path (str): xmls保存的文件夹路径
        src_images_path (str): xmls对应的图片文件夹路径
        src_images_type (tuple, optional): 图片的格式. Defaults to ImageFormat.
        verbose (bool, optional): 日志是否显示在终端. Defaults to False.
        confirm (bool, optional): 是否进行参数确认. Defaults to True.
    """

    # 获取xml文件列表
    annotation_files = [file for file in os.listdir(src_xmls_path) if file.lower().endswith('.xml')]

    "------------计数------------"
    count_total_xml   = len(annotation_files)  # 需要处理的 .xml 文件数量
    count_succeed     = 0  # 成功修改的数量
    count_skip        = 0  # 跳过的数量
    count_error       = 0  # 错误的数量
    "---------------------------"
    
    # 获取日志
    logger = get_logger(verbose=verbose)
    lsp = get_logger_save_path(logger)
    
    _str = [
        ['xml文件夹路径', src_xmls_path],
        ['需要处理的xml文件数量', count_total_xml],
        ['xml文件夹保存路径', dst_xmls_save_path],
        ['xml对应的图片文件夹路径', src_images_path],
        ['xml对应的图片类型', src_images_type],
        ['日志保存路径', lsp],
    ]
    
    _str = print_arguments(params_dict=_str, confirm=confirm)
    logger.info(f"\n{_str}")

    # 遍历所有的xml文件
    process_bar = tqdm(total=count_total_xml, desc="根据图片修正 xml 文件的尺寸 <size> 信息", unit='xml')
    for xml_file in annotation_files:
        xml_name, _ = os.path.splitext(xml_file)  # 分离文件名和后缀
        process_bar.set_description(f"Process in \033[1;31m{xml_file}\033[0m")
        
        # 读取 xml 文件
        xml_path = os.path.join(src_xmls_path, xml_file)  # 获取完整路径
        tree     = ET.parse(xml_path)  # 解析 xml 树
        root     = tree.getroot()  # 获取 xml 树的根
        
        # 获取同名图片文件名
        image_path = os.path.join(src_images_path, xml_name) + src_images_type
        
        # 判断对应的同名图片文件是否存在，如果不存在则记录错误
        if not os.path.exists(image_path):
            logger.error(f"Error: {xml_path} -> The corresponding image doesn't existed!")
            count_error += 1
            process_bar.update()
            continue
        
        # 使用PIL获取图片尺寸
        image = Image.open(image_path)
        width, height = image.size
        
        # 判断 xml 中的 <size> 标签是否和图片尺寸对应
        size_elem = root.find("size")
        if size_elem.find("width").text == str(width) and size_elem.find("height").text == str(height):
            # 不需要修正，直接保存文件
            output_path = os.path.join(dst_xmls_save_path, xml_file)
            tree.write(output_path, encoding="utf-8")
            count_skip += 1
            logger.info(f"Skip: {xml_path} -> Don't need modify.")
            process_bar.update()
            continue
        else:
            old_width, old_height = size_elem.find("width").text, size_elem.find("height").text
            
            # 更新xml中的<size>标签
            size_elem.find("width").text = str(width)
            size_elem.find("height").text = str(height)

            # 保存修正后的xml文件
            output_path = os.path.join(dst_xmls_save_path, xml_file)
            tree.write(output_path, encoding="utf-8")
            logger.info(f"Processed: {xml_path} -> ({old_width}, {old_height}) -> "
                        f"({size_elem.find('width').text}, {size_elem.find('height').text})")
            count_succeed += 1
            process_bar.update()
    process_bar.close()

    _str = [
        ["成功修正数量/总xml数量", f"{count_succeed}/{count_total_xml}"],
        ["跳过数量/总xml数量", f"{count_skip}/{count_total_xml}"],
        ["出错数量/总xml数量", f"{count_error}/{count_total_xml}"],
    ]
    
    _str = print_arguments(params_dict=_str)
    logger.info(f"\n{_str}")

    if count_error == 0:
        _str = "👌  No Problems!"
        xprint(_str, color='green', bold=True, hl='>', hl_num=2)
        logger.info(_str)
    else:
        _str = "🤡  有问题, 请仔细核查!"
        xprint(_str, color='red', bold=True, hl='>')
        
        _str = [
            ["成功量", count_succeed],
            ["跳过量", count_skip],
            ["错误量", count_error],
            ["成功+跳过+错误", count_succeed + count_skip + count_error],
            ["xml总量", count_total_xml],
        ]
        
        _str = print_arguments(params_dict=_str)
        logger.info(f"\n{_str}")
        

if __name__ == "__main__":
    check_xml_size_info(
        src_xmls_path='utils/dataset/EXAMPLE_FOLDER/annotations',
        src_images_path='utils/dataset/EXAMPLE_FOLDER/images',
        dst_xmls_save_path='utils/dataset/EXAMPLE_FOLDER/annotations-xml-recheck',
    )