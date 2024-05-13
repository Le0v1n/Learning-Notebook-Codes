import os
import sys
sys.path.append(os.getcwd())
import cv2
from utils.augmentations import letterbox


if __name__ == "__main__":
    # 图片想要变为的尺寸
    dst_size = (416, 416)
    
    # 读取图片
    # img = cv2.imread('examples/img-terminator.png')
    img = cv2.imread('examples/img-doraemon.png')
    
    # 直接resize
    img_resize = cv2.resize(img, dsize=dst_size)
    cv2.imwrite("examples/resize.png", img_resize)
    
    # 使用LetterBox对图片进行裁剪
    img_letterbox, ratio, padding = letterbox(
        im=img,
        new_shape=dst_size,
        auto=False,  # 如果auto=True，则为rectangle（宽度或高度一个边进行填充，填充到stride的最小倍数即可 -> 得到的是一个矩形）。如果auto=False，则为squared（宽度或高度一个边进行填充，填充到目标尺寸 -> 得到的是一个正方形）
        scaleFill=False,
        scaleup=True,  # 如果图片比较小，scaleup=False，则不会对图片进行resize，而是在四周进行Padding
        stride=32
    )
    
    desc = "由大变小" if img.shape[1] > dst_size[0] or img.shape[0] > dst_size[1] else "由小变大"
    print(f"---------------------------------------- INFO ----------------------------------------\n"
          f"[{desc}] 原始图片尺寸为：{img.shape[:2]}\n"
          f"[{desc}] 原始宽高/目标宽高的最小值：{ratio}\n"
          f"[{desc}] 宽度和高度的填充数（实际应该x2）：{padding}\n"
          f"--------------------------------------------------------------------------------------")
    cv2.imwrite("examples/letterbox.png", img_letterbox)