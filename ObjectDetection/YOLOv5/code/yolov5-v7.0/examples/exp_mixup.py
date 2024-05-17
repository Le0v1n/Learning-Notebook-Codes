import os
import sys
sys.path.append(os.getcwd())
import cv2
import numpy as np
from utils.augmentations import letterbox, mixup


# 创建一个函数来将图片的名称转换为标签
def create_label_from_name(image_name, class_names):
    # 假设图片名称中包含其内容的描述，例如 "img-2person-1robot.png"
    labels = [0] * num_classes
    for part in image_name.split('-')[1:]:  # 跳过文件名的前缀，如"img"
        if part:
            count, class_name = part[0], part[1:]
            if class_name in class_names:
                index = class_names.index(class_name)
                labels[index] = int(count)  # 将字符串转换为整数
    return np.array(labels)


if __name__ == "__main__":
    # 定义类别及其索引
    class_names = ['person', 'robot', 'fairy', 'planet', 'spaceship']
    num_classes = len(class_names)
    
    # 读取图片
    img_1 = cv2.imread('examples/img-terminator.png')
    img_2 = cv2.imread('examples/img-doraemon.png')
    
    # 创建标签
    label_1 = np.array([2, 0, 0, 0, 0])
    label_2 = np.array([4, 1, 1, 1, 0])
    print(f"{label_1.shape = }")
    print(f"{label_2.shape = }")
    
    # 首先使用letterbox的square将图片弄到相同的尺寸
    img_1 = letterbox(img_1, auto=False)[0]
    img_2 = letterbox(img_2, auto=False)[0]
    
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