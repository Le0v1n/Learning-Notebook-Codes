import os
import sys
import numpy as np
import cv2
import json
from tqdm.rich import tqdm

sys.path.append(os.getcwd())
from utils.common_fn import xprint, print_arguments, get_logger, get_logger_save_path
from utils.file_type import ImageFormat


__doc__ = """脚本说明：为负样本生成空的json文件，如果保存目录中有json文件则不创建（确保正样本的json文件不会被覆盖）
    用途：为负样本生成空的json文件
    要求：要有对应的图片文件，这样读取到的尺寸信息是最准确的。
    注意: 无。
"""

def create_empty_json(src_images_path: str, 
                      dst_json_path: str, 
                      src_image_type: tuple = ImageFormat, 
                      dst_json_version: str = "0.2.2",
                      verbose=False, confirm=True):

    # Json 文件基础信息
    __version = dst_json_version
    __flags = {},
    __shapes = [],
    __imageData = None

    # 获取所有图片
    image_list = [file for file in os.listdir(src_images_path) if file.endswith(src_image_type)]
    json_list = [file for file in os.listdir(dst_json_path) if file.endswith('.json')]

    "------------计数------------"
    count_total = len(image_list)
    count_succeed = 0
    count_skip = 0
    "---------------------------"
    
    # 打开日志
    logger = get_logger(verbose=verbose)
    lsp = get_logger_save_path(logger)
    
    _str = [
        ['图片路径', src_images_path],
        ['json路径', dst_json_path],
        ['json版本', dst_json_version],
        ['图片数量', count_total],
        ['目前json数量', len(json_list)],
        ['预计生成的json数量', count_total - len(json_list)],
        ['日志文件路径', lsp]
    ]
    _str = print_arguments(params_dict=_str, confirm=confirm)
    logger.info(f"\n{_str}")

    # 创建进度条
    progress_bar = tqdm(total=count_total, desc="为负样本创建json文件", unit=" .json")
    for image_name in image_list:
        progress_bar.set_description(f"Process in \033[1;31m{image_name}\033[0m")
        image_pre, _ = os.path.splitext(image_name)  # 分离文件名和后缀

        # 保存的路径
        json_sp = os.path.join(dst_json_path, image_pre) + '.json'
        # 图片的路径
        img_rp = os.path.join(src_images_path, image_name)

        # 判断对应的json文件是否存在
        if os.path.exists(json_sp):
            count_skip += 1
            logger.info(f"Skip: {json_sp} has existed.")
            progress_bar.update(1)
            continue

        # 读取图片获取尺寸信息
        img = cv2.imdecode(np.fromfile(img_rp, dtype=np.uint8), cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        content = {"version": __version,
                   "flags": __flags,
                   "shapes": __shapes,
                   "imagePath": f"{image_pre}.jpg",
                   "imageData": __imageData,
                   "imageHeight": height,
                   "imageWidth": width
                   }

        # 创建json文件并写入内容
        with open(json_sp, 'w') as f:
            json.dump(content, f, indent=2)
        
        count_succeed += 1
        logger.info(f"Succeed: {json_sp}")
        progress_bar.update(1)
    progress_bar.close()
    
    _str = "为负样本创建json文件已完成，详情如下"
    xprint(_str, color='green', bold=True, hl='>')
    logger.info(_str)
    
    _str = [
        ['👌成功', f"{count_succeed}/{count_total}"],
        ['👌跳过', f"{count_skip}/{count_total}"],
    ]
    
    _str = print_arguments(params_dict=_str)
    logger.info(f"\n{_str}")

    if count_succeed + count_skip == count_total and count_succeed == count_total - len(json_list):
        _str = "👌 No Problems"
        xprint(_str, color='green', bold=True, hl='>', hl_num=2, hl_style='full')
        logger.info(_str)


if __name__ == "__main__":
    create_empty_json(
        src_images_path='utils/dataset/EXAMPLE_FOLDER/images',
        dst_json_path='utils/dataset/EXAMPLE_FOLDER/annotations',
        src_image_type='.jpg'
    )