import os


dataset_path = 'Datasets/coco128/train/images'
image_names = ['1.jpg', '2.jpg', '3.jpg', '4.png', '5.jpeg', '6.gif']

print(f"---------- 使用 os.path.join(path1, path2, ...) 拼接路径 ----------")
image_paths = [os.path.join(dataset_path, image_name) for image_name in image_names 
               if image_name.endswith(('.jpg', '.png'))]
[print(f"{image_path = }") for image_path in image_paths]

print(f"\n---------- 使用 str.join(seq) 拼接路径 ----------")
image_paths = [dataset_path.join(image_name) for image_name in image_names 
               if image_name.endswith(('.jpg', '.png'))]
[print(f"{image_path = }") for image_path in image_paths]