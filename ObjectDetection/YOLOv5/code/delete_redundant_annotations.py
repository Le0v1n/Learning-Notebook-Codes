import os
from tqdm import tqdm


# 定义图片文件夹和标签文件夹的路径
images_folder = '/mnt/c/Users/Le0v1n/Desktop/测试案例/Datasets/exp_1/JPEGImages'
annotations_folder = '/mnt/c/Users/Le0v1n/Desktop/测试案例/Datasets/exp_1/Annotations'

# 获取images文件夹中的所有图片文件
image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')]

# 获取annotations文件夹中的所有.xml文件
annotation_files = [f for f in os.listdir(annotations_folder) if f.endswith('.xml')]

if len(image_files) == len(annotation_files):
    print(f"两种文件夹中文件数量相同({len(image_files)} v.s. {len(annotation_files)})，程序退出!")
    exit()

# 获取images文件夹中存在的图片文件的文件名（不包含扩展名）
existing_image_names = set(os.path.splitext(f)[0] for f in image_files)

# 使用tqdm创建进度条
deleted_num = 0
with tqdm(total=len(annotation_files), desc="删除标签文件进度") as pbar:
    # 遍历annotations文件夹，删除没有对应图片的.xml文件
    for annotation_file in annotation_files:
        annotation_name = os.path.splitext(annotation_file)[0]

        if annotation_name not in existing_image_names:
            # 构建要删除的.xml文件的完整路径
            annotation_path = os.path.join(annotations_folder, annotation_file)
            # 删除文件
            os.remove(annotation_path)
            pbar.update(1)  # 更新进度条
            pbar.set_postfix(deleted=annotation_file)  # 显示已删除的文件名
            deleted_num += 1

print(f"删除操作完成, 共删除 {deleted_num} 个 .xml 文件")

# 再检查一遍
# 获取images文件夹中的所有图片文件
image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')]

# 获取annotations文件夹中的所有.xml文件
annotation_files = [f for f in os.listdir(annotations_folder) if f.endswith('.xml')]

if len(image_files) == len(annotation_files):
    print(f"两种文件夹中文件数量相同({len(image_files)} v.s. {len(annotation_files)})，程序退出!")
else:
    print(f"两个文件夹数量不相同({len(image_files)} v.s. {len(annotation_files)})，可能存在纯负样本!")
