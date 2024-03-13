import os
import sys
import json
import cv2
import numpy as np
from typing import Union
from collections.abc import Iterable
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
    

def greyscale_to_labelme_json(images_dir: str, 
                              annotations_dir: str, 
                              jsons_save_path: str, 
                              class_map: Union[dict, tuple, list],
                              fitting_accuracy='normal') -> None:
    """Convert grey annotations (.png) into .json for labelme

    Args:
        images_dir (str): The path of images dir
        annotations_dir (str): The path of annotation dir
        jsons_save_path (str): The path of saving results (.json)
        class_map (Union[dict, tuple, list]): The dict of class names. eg. {0: 'bg', 1: 'cls1'}. 
                                              And you can use list or tuple like ('bg', 'cls1', ...).
        fitting_accuracy (str, optional): The accuracy of fitting the contours. Defaults to 'normal'.
                                          Options: bad, normal, good.
    """
    # Convert tuple/list into dict like {0: 'background', 1: 'cls1', ...}
    if isinstance(class_map, (tuple, list)):
        class_map = {idx: c for idx, c in enumerate(class_map)}
        
    # Check background
    if class_map[0].lower() not in ('bg', 'back', 'background'):
        xprint(f"⚠️  class_map 中第一个类别 {class_map[0]} 不在 ['bg', 'back', 'background'] 中，请确认是否继续!", 
               color='red', hl='>', hl_style='full', hl_num=2, bold=True)
        print_arguments(only_wait=True)
        
    # convert fitting_accuracy
    if fitting_accuracy.lower() in ('normal', 'default'):
        epsilon = 1
    elif fitting_accuracy.lower() in ('best', 'big', 'more', 'perfect', 'good'):
        epsilon = 0
    elif fitting_accuracy.lower() in ('worst', 'small', 'less', 'bad'):
        epsilon = 4

    # Get the images and annotation list
    images_list = [file for file in os.listdir(images_dir) if file.lower().endswith(ImageFormat)]
    
    # create save folder
    create_folder(jsons_save_path)
    
    # tqdm
    process_bar = tqdm(total=len(images_list), desc='png2json')
    
    # ptab
    ptab = PrettyTable(['No', 'Error Type', 'Path'])

    for image_name in images_list:
        process_bar.set_description(f"Processing {image_name}")
        
        # Paths
        pre, ext = os.path.splitext(image_name)
        image_path = os.path.join(images_dir, image_name)
        annotation_path = os.path.join(annotations_dir, pre) + '.png'  # assume
        json_save_path = os.path.join(jsons_save_path, pre) + '.json'  # assume
        
        # Confirm image and annotation exist
        if not os.path.exists(image_path):
            xprint(f"❌  {image_path} doesn't exist!")
            ptab.add_row([f"{len(ptab.rows)+1}", 'Not Exist', image_path])  # report error
            process_bar.update()
            continue
        elif not os.path.exists(annotation_path):
            xprint(f"❌  {annotation_path} doesn't exist!")
            ptab.add_row([f"{len(ptab.rows)+1}", 'Not Exist', annotation_path])  # report error
            process_bar.update()
            continue

        # Load image and annotation
        image = cv2.imread(image_path)
        annotation = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)

        # Get the size of annotation_img
        image_height, image_width, _ = image.shape
        annotation_height, annotation_width = annotation.shape
        
        if image_height != annotation_height or image_width != annotation_width:
            xprint(f"❌  The sizes of image and annotation don't match!\n"
                   f"\timage_height: {image_height}\tannotation_height: {annotation_height}\n"
                   f"\timage_width: {image_width}\tannotation_width: {annotation_width}")
            ptab.add_row([f"{len(ptab.rows)+1}", 'Sizes', image_path])  # report error
            process_bar.update()
            continue

        # Create labelme JSON structure
        labelme_data = {
            "version": "5.2.1",
            "flags": {},
            "shapes": [],
            "imagePath": image_name,  # relative path
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width
        }

        # Iterate through category map and create labelme shapes
        for class_id, class_name in class_map.items():
            mask = (annotation == class_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Approximate contour to reduce points
                approx_contour = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
                points = approx_contour.squeeze().tolist()
                
                # Don't record single point
                if not isinstance(points[0], Iterable):
                    continue
                
                # Check if the contour is not empty
                if len(approx_contour) > 0:
                    shape = {
                        "label": class_name,
                        "points": points,
                        "group_id": None,
                        "description": "",
                        "shape_type": "polygon",
                        "flags": {}
                    }
                    labelme_data["shapes"].append(shape)

        # Save labelme JSON file
        with open(json_save_path, 'w') as json_file:
            json.dump(labelme_data, json_file, indent=2)
            
        process_bar.update()
    process_bar.close()
    
    print(ptab) if len(ptab.rows) > 0 else ...
    xprint(f"The results have been saved in {jsons_save_path}", color='blue', hl='>', bold=True)
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, help='Dataset image directory')
    parser.add_argument('--labels-dir', type=str, help='Dataset labels directory')
    parser.add_argument('--save-dir', type=str, help='Save path of results (.png)')
    parser.add_argument('--class_name', type=str, nargs='+', help='The class names, e.g. --class_name bg cat dog')
    parser.add_argument('--fitting-accuracy', type=str, default='normal', help='The accuracy of fitting')
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse_args()
    
    greyscale_to_labelme_json(args.images_dir, 
                              args.labels_dir, 
                              args.save_dir, 
                              class_map=args.class_name,
                              fitting_accuracy=args.fitting_accuracy)
