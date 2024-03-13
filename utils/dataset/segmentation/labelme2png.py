import os
import sys
import json
from PIL import Image, ImageDraw
from typing import Union
from prettytable import PrettyTable
import argparse

sys.path.append(os.getcwd())

from utils.outer import print_arguments, xprint
from utils.generator import create_folder
from utils.items import ImageFormat

try:
    from tqdm.rich import tqdm
except:
    from tqdm import tqdm


# 这个函数将Labelme的JSON标签转换为PNG标签
def json2png(images_dir: str, 
                jsons_dir: str, 
                png_save_dir: str, 
                class_map: Union[dict, list, tuple]) -> None:
    
    # Convert tuple/list into dict like {'background': 0, 'cls1': 1, ...}
    if isinstance(class_map, (tuple, list)):
        class_map = {c: idx for idx, c in enumerate(class_map)}
        
    # Check background
    if list(class_map.keys())[0].lower() not in ('bg', 'back', 'background'):
        xprint(f"⚠️  class_map 中第一个类别 {list(class_map.keys())[0]} "
               f"不在 ['bg', 'back', 'background'] 中，请确认是否继续!", 
               color='red', hl='>', hl_style='full', hl_num=2, bold=True)
        print_arguments(only_wait=True)
    
    # Get the images list
    images_list = [file for file in os.listdir(images_dir) if file.lower().endswith(ImageFormat)]
    
    # create save folder
    create_folder(png_save_dir)
    
    # tqdm
    process_bar = tqdm(total=len(images_list), desc='json2png', unit='json')
    
    # ptab
    ptab = PrettyTable(['No', 'Error Type', 'Path'])
    
    for image_name in images_list:
        process_bar.set_description(f"Processing {image_name}")
        
        # Paths
        pre, ext = os.path.splitext(image_name)
        image_path = os.path.join(images_dir, image_name)
        json_path = os.path.join(jsons_dir, pre) + '.json'  # assume
        png_save_path = os.path.join(png_save_dir, pre) + '.png'  # assume
        
        # Confirm image and json exist
        if not os.path.exists(image_path):
            xprint(f"❌  {image_path} doesn't exist!")
            ptab.add_row([f"{len(ptab.rows)+1}", 'Not Exist', image_path])  # report error
            process_bar.update()
            continue
        elif not os.path.exists(json_path):
            xprint(f"❌  {json_path} doesn't exist!")
            ptab.add_row([f"{len(ptab.rows)+1}", 'Not Exist', json_path])  # report error
            process_bar.update()
            continue
        
        # Load image and json
        image = Image.open(image_path)
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            
        # Check the size of image and annotation
        image_height, image_width = image.size
        annotation_height, annotation_width = json_data['imageWidth'], json_data['imageHeight']

        if image_height != annotation_height or image_width != annotation_width:
            xprint(f"❌  The sizes of image and annotation don't match!\n"
                   f"\timage_height: {image_height}\tannotation_height: {annotation_height}\n"
                   f"\timage_width: {image_width}\tannotation_width: {annotation_width}")
            ptab.add_row([f"{len(ptab.rows)+1}", 'Sizes', image_path])  # report error
            process_bar.update()
            continue

        # 创建一个与原始图像大小相同的单通道图像
        semantic_img = Image.new('L', (annotation_height, annotation_width), 0)

        # 创建一个可以在图像上绘图的对象
        draw = ImageDraw.Draw(semantic_img)

        # 遍历每个对象
        for shape in json_data['shapes']:
            # 获取对象的标签（类别）
            class_name = shape['label']

            # 使用类别字典将类别名称转换为整数标签
            label_value = class_map.get(class_name, 0)  # 如果类别未知，则默认为背景（0）

            # 获取对象的多边形点
            points = shape['points']

            # 将点转换为适合PIL的格式
            vertices = [(p[0], p[1]) for p in points]
            
            if len(vertices) < 2:
                process_bar.update()
                continue

            # 填充多边形
            draw.polygon(vertices, fill=label_value)

        # 保存图像
        semantic_img.save(png_save_path)
        process_bar.update()
    process_bar.close()
    
    print(ptab) if len(ptab.rows) > 0 else ...
    xprint(f"The results have been saved in {png_save_dir}", color='blue', hl='>', bold=True)
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, help='Dataset image directory')
    parser.add_argument('--labels-dir', type=str, help='Dataset labels directory')
    parser.add_argument('--save-dir', type=str, help='Save path of results (.png)')
    parser.add_argument('--class_name', type=str, nargs='+', help='The class names, e.g. --class_name cat dog lion')
    args = parser.parse_args()
    
    return args


if __name__  == "__main__":
    args = parse_args()
    
    json2png(images_dir=args.images_dir, 
             jsons_dir=args.labels_dir,
             png_save_dir=args.save_dir, 
             class_map=args.class_name)
