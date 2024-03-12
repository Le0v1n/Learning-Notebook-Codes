import os
import sys
from tqdm.rich import tqdm
import shutil
from PIL import Image
import argparse
from utils.generator import create_folder

from utils.outer import print_arguments, xprint

sys.path.append(os.getcwd())
from utils.getter import get_logger
from utils.items import ImageFormat


__doc__ = """脚本说明：对指定文件夹下所有的图片进行格式转换
    用途：统一数据集图片的格式
    要求：无
    注意：
        1. 不需要转换的则跳过
        2. 不是图片的文件有两种操作方式：
          2.1 mv/move 扔到 RECYCLE_BIN_PATH
          2.2 del/delete 直接删除
"""


def convert_images_format(src_image_path, src_image_type=ImageFormat, dst_image_type='.jpg', 
                          operation_method='mv', verbose=True, confirm=True):
    """对指定文件夹下所有的图片进行格式转换

    Args:
        src_image_path (str): 图片文件夹路径
        src_image_type (tupe, optional): 图片的格式 -> (".jpg", ".png", ...). Defaults to ImageFormat.
        dst_image_type (str, optional): 想要转换为什么格式. Defaults to '.jpg'.
        operation_method (str, optional): 操作方式 -> 'mv', 'del'. Defaults to 'mv'.
                                          mv: 移动到垃圾桶中
                                          del: 直接删除
        verbose (bool, optional): 日志是否显示在终端. Defaults to True.
        confirm (bool, optional): 是否检查参数. Defaults to False.
    """

    if dst_image_type == '.jpg':
        _convert_type_in_pil = 'JPEG'
    elif dst_image_type == '.png':
        _convert_type_in_pil = 'PNG'
    elif dst_image_type == '.gif':
        _convert_type_in_pil = 'GIF'
    else:
        raise KeyError("⚠️  只能转换为 [.jpg, .png, .gif]")

    # 获取所有图片和标签
    images_list = os.listdir(src_image_path)

    # 过滤只包括特定类型的图像文件
    images_list = [file for file in images_list if file.lower().endswith(src_image_type)]  # 需要转换的图片list
    dont_convert_images_list = [file for file in images_list if file.lower().endswith(dst_image_type)]  # 不需要转换的图片list

    "------------计数------------"
    count_total_images = len(images_list)
    count_need_convert = len(dont_convert_images_list)
    count_succeed = 0
    count_skip = 0
    "---------------------------"

    del dont_convert_images_list  # 删除list释放空间

    if count_need_convert == count_total_images:  # 如果没有要转换的图片
        xprint(f"💡  所有图片均为 {dst_image_type}({count_need_convert}张), 无需转换, 程序停止!", 
               color='blue', hl=">", hl_style='full')
        exit()

    # 创建logger
    logger = get_logger(verbose=verbose)
    lsp = logger.handlers[0].baseFilename  # logging_save_path
    lsp = os.path.relpath(lsp, os.getcwd())

    _str = [
        ['图片路径', src_image_path],
        ['转换格式', dst_image_type],
        ['操作方式', operation_method],
        ['所有图片数量', count_total_images],
        [f'已有{dst_image_type} 图片数量', count_need_convert],
        ['预计转换图片数量', count_total_images - count_need_convert],
        ['日志保存路径', lsp],
    ]
    
    _str = print_arguments(params_dict=_str, confirm=confirm)
    logger.info(f"\n{_str}")
        
    if operation_method in ('mv', 'move'):
        recycle_bin_path = os.path.join(os.path.dirname(src_image_path), "Recycle_bins")  # 将垃圾桶放在图片的上一级目录中
        create_folder(fp=recycle_bin_path)

    # 创建进度条
    progress_bar = tqdm(total=count_total_images, desc=f"Convert {src_image_type} into {dst_image_type}", unit="img")
    for image_name in images_list:
        pre, ext = os.path.splitext(image_name)  # 分离文件名和后缀

        # 如果是我们想要的图片格式则跳过
        if ext == dst_image_type:
            count_skip += 1
            progress_bar.update()
            continue
        
        # 需要转换
        image_path = os.path.join(src_image_path, image_name)  # 被转换图片的完整路径
        image_save_path = os.path.join(src_image_path, pre) + dst_image_type  # 转换后图片的保存路径
        
        # 打开图片并进行转换
        image = Image.open(image_path)
        image.save(image_save_path, _convert_type_in_pil)
        
        # 开始操作原有的图片: 移到垃圾桶还是直接删除
        if operation_method in ('mv', 'move'):
            dst_path = os.path.join(recycle_bin_path, image_name)
            shutil.move(src=image_path, dst=dst_path)
        elif operation_method in ('del', 'delete'):
            os.remove(image_path)
        
        # 记录日志
        if dst_path:
            logger.info(msg=f"{image_path} -> {image_save_path}\t{dst_path}")
        else:
            logger.info(msg=f"{image_path} -> {image_save_path}")
                    
        count_succeed += 1
        progress_bar.update(1)
    progress_bar.close()
    
    _str = [
        ['成功转换的图片数量', f"{count_succeed}/{count_total_images}"],
        ['跳过转换的图片数量', f"{count_skip}/{count_total_images}"],
        ['转换后的图片路径', src_image_path],
        ['日志保存路径', lsp],
    ]
    _str.append(['垃圾桶路径', recycle_bin_path]) if operation_method in ('mv', 'move') else ...
    
    # 打印结果
    _str = print_arguments(params_dict=_str)
    logger.info(f"\n{_str}")
    
    if count_succeed + count_skip == count_total_images and count_skip == count_need_convert:
        xprint(f"👌 No problems", color='green', bold=True, hl='>')


if __name__ == "__main__":
    convert_images_format(src_image_path='utils/dataset/EXAMPLE_FOLDER/images',
                          src_image_type=('.jpg', '.png', '.jpeg'),
                          dst_image_type='.jpg',
                          operation_method='mv')