import os
import sys
sys.path.append(os.getcwd())
import cv2
import numpy as np
from utils.augmentations import letterbox, mixup


def get_label(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    # 分行
    split_lines = [line.split() for line in lines]  
    
    # 转换为ndarray
    label = np.array(split_lines, dtype=np.float32)
    
    return label

if __name__ == "__main__":
    # 定义类别及其索引
    class_names = ['person', 'robot', 'fairy', 'planet', 'spaceship']
    num_classes = len(class_names)
    
    path_img_1 = 'examples/000000000081.jpg'
    path_img_2 = 'examples/000000000572.jpg'
    path_label_1 = 'examples/000000000081.txt'
    path_label_2 = 'examples/000000000572.txt'
    
    # 读取图片
    img_1 = cv2.imread(path_img_1)
    img_2 = cv2.imread(path_img_2)
    
    # 首先使用letterbox的square将图片弄到相同的尺寸
    img_1, ratio_1, pad_1 = letterbox(img_1, auto=False)
    img_2, ratio_2, pad_2 = letterbox(img_2, auto=False)

    
    # 读取标签
    label_1 = get_label(path_label_1)
    label_2 = get_label(path_label_2)
    
    
    # 应用MixUp增强
    mixed_img, mixed_labels = mixup(
        im=img_1,
        labels=label_1,
        im2=img_2,
        labels2=label_2
    )
    
    # 显示混合后的图像
    cv2.imwrite('examples/mix_up.png', mixed_img)
    
    # 打印混合后的标签
    print("Mixed Labels:", mixed_labels)
    print(f"{mixed_labels.shape = }")