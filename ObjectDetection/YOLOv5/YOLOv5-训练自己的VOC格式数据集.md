<center><b><font size=12>YOLOv5：训练自己的 VOC 格式数据集</font></b></center>

# 1. 自定义数据集

## 1.1 环境安装

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**注意**：
1. 安装 `lxml`
2. Pillow 版本要低于 10.0.0，解释链接: [module 'PIL.Image' has no attribute 'ANTIALIAS' 问题处理](https://baijiahao.baidu.com/s?id=1775432196700665405)

## 1.2 创建数据集

我们自己下载 PASCAL VOC 也行，按照 PASCAL VOC 自建一个也行，具体过程见 [PASCAL VOC 2012数据集讲解与制作自己的数据集](https://blog.csdn.net/weixin_44878336/article/details/124540069)。

> 文章不长

## 1.3 PASCAL VOC 数据集结构

PASCAL VOC 数据集结构如下所示。

```txt
PASCAL VOC 2012 数据集
|
├── VOC2012
|   ├── JPEGImages    # 包含所有图像文件
|   |   ├── 2007_000027.jpg
|   |   ├── 2007_000032.jpg
|   |   ├── ...
|   |
|   ├── Annotations    # 包含所有标注文件（XML格式）
|   |   ├── 2007_000027.xml
|   |   ├── 2007_000032.xml
|   |   ├── ...
|   |
|   ├── ImageSets
|   |   ├── Main
|   |   |   ├── train.txt  # 训练集的图像文件列表
|   |   |   ├── val.txt    # 验证集的图像文件列表
|   |   |   ├── test.txt   # 测试集的图像文件列表
|   |
|   ├── SegmentationClass  # 语义分割的标注
|   |   ├── 2007_000032.png
|   |   ├── ...
|   |
|   ├── SegmentationObject  # 物体分割的标注
|   |   ├── 2007_000032.png
|   |   ├── ...
|   |
|   ├── ...               # 其他可能的子文件夹
|
├── VOCdevkit
|   ├── VOCcode          # 包含用于处理数据集的工具代码
|
├── README
```

我们可以看到，对于我们来说，我们只需要两个文件夹就可以了。

1. JPEGImages: 存放所有的图片
2. Annotations: 存放所有的标注信息

这里我们从 PASCAL VOC 中提取出几张图片，组成 VOC2012-Lite：

<div align=center>
    <img src=./imgs_markdown/2023-10-18-17-19-28.png
    width=30%>
</div>

即此时我们的数据集结构为：

```txt
VOCdevkit
└─VOC2012-Lite
    ├─Annotations
    │      2007_000027.xml
    │      2007_000032.xml
    │      2007_000033.xml
    │      2007_000039.xml
    │      2007_000042.xml
    │      2007_000061.xml
    │      ...
    │
    └─JPEGImages
            2007_000027.jpg
            2007_000032.jpg
            2007_000033.jpg
            2007_000039.jpg
            2007_000042.jpg
            2007_000061.jpg
            ...
```

需要注意的是，YOLOv5 的要求标注文件后缀为 `.txt`，但 Annotations 中的文件后缀是 `.xml`，所以我们需要进行转换。

<details>
<summary>YOLO 标注文件说明 </summary>

标注文件举例：

```txt
0 0.481719 0.634028 0.690625 0.713278
1 0.741094 0.524306 0.314750 0.933389
2 0.254162 0.247742 0.574520 0.687422
```

其中，每行代表一个物体的标注，每个标注包括五个值，分别是：

1. `<class_id>`：物体的类别标识符。在这里，有三个不同的类别，分别用 0、1 和 2 表示。
2. `<center_x>`：物体边界框的中心点 x 坐标，归一化到图像宽度。这些值的范围应在 0 到 1 之间。
3. `<center_y>`：物体边界框的中心点 y 坐标，归一化到图像高度。同样，这些值的范围应在 0 到 1 之间。
4. `<width>`：物体边界框的宽度，归一化到图像宽度。
5. `<height>`：物体边界框的高度，归一化到图像高度。

以第一行为例：

- `<class_id>` 是 0，表示这个物体属于类别 0。
- `<center_x>` 是 0.481719，这意味着物体边界框的中心点 x 坐标位于图像宽度的 48.17% 处。
- `<center_y>` 是 0.634028，中心点 y 坐标位于图像高度的 63.40% 处。
- `<width>` 是 0.690625，边界框宽度占图像宽度的 69.06%。
- `<height>` 是 0.713278，边界框高度占图像高度的 71.33%。

</details>

## 1.4 YOLO 想要的数据集结构

### 1.4.1 YOLOv3

一般而言，YOLOv3 想要的数据结构如下所示：

```txt
YOLOv3 数据集
|
├── images         # 包含所有图像文件
|   ├── image1.jpg
|   ├── image2.jpg
|   ├── ...
|
├── labels         # 包含所有标注文件（每个图像对应一个标注文件）
|   ├── image1.txt
|   ├── image2.txt
|   ├── ...
|
├── classes.names  # 类别文件，包含所有类别的名称
|
├── train.txt      # 训练集的图像文件列表
├── valid.txt      # 验证集的图像文件列表
```

### 1.4.2 YOLOv5

与 YOLOv3 不同，YOLOv5 所需要的数据集结构如下所示：

```txt
|-- test
|   |-- images
|   |   |-- 000000000036.jpg
|   |   `-- 000000000042.jpg
|   `-- labels
|       |-- 000000000036.txt
|       `-- 000000000042.txt
|-- train
|   |-- images
|   |   |-- 000000000009.jpg
|   |   `-- 000000000025.jpg
|   `-- labels
|       |-- 000000000009.txt
|       `-- 000000000025.txt
`-- val
    |-- images
    |   |-- 000000000030.jpg
    |   `-- 000000000034.jpg
    `-- labels
        |-- 000000000030.txt
        `-- 000000000034.txt
```

既然我们已经知道了 YOLOv5 所需要的数据集格式，那么就可以动手了！

## 1.5 将 PASCAL VOC 数据集转换为 YOLOv5 数据集格式

<details><summary>voc2yolo.py</summary>

```python
"""
本脚本有两个功能：
    1. 将 voc 数据集标注信息(.xml)转为 yolo 标注格式(.txt)，并将图像文件复制到相应文件夹
    2. 根据 json 标签文件，生成对应 names 标签(my_data_label.names)
    3. 兼容 YOLOv3 和 YOLOv5
"""
import os
from tqdm import tqdm
from lxml import etree
import json
import shutil
import argparse
from tqdm import tqdm
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split


def args_table(args):
    # 创建一个表格
    table = PrettyTable(["Parameter", "Value"])
    table.align["Parameter"] = "l"  # 使用 "l" 表示左对齐
    table.align["Value"] = "l"  # 使用 "l" 表示左对齐

    # 将args对象的键值对添加到表格中
    for key, value in vars(args).items():
        # 处理列表的特殊格式
        if isinstance(value, list):
            value = ', '.join(map(str, value))
        table.add_row([key, value])

    # 返回表格的字符串表示
    return str(table)


def generate_train_and_val_txt(args):
    
    target_train_file = args.train_txt_path
    target_val_file = args.val_txt_path

    # 获取源文件夹中的所有文件
    files = os.listdir(args.voc_images_path)
    
    # 划分训练集和验证集
    train_images, val_images = train_test_split(files, test_size=args.val_size, random_state=args.seed)

    # 打开目标文件以写入模式
    with open(target_train_file, 'w', encoding='utf-8') as f:
        # 使用tqdm创建一个进度条，迭代源文件列表
        for file in tqdm(train_images, desc=f"\033[1;33mProcessing Files for train\033[0m"):
            file_name, _ = os.path.splitext(file)
            # 写入文件名
            f.write(f'{file_name}\n')

    with open(target_val_file, 'w', encoding='utf-8') as f:
        # 使用tqdm创建一个进度条，迭代源文件列表
        for file in tqdm(val_images, desc=f"\033[1;33mProcessing Files for val\033[0m"):
            file_name, _ = os.path.splitext(file)
            # 写入文件名
            f.write(f'{file_name}\n')

    print(f"\033[1;32m文件名已写入到 {target_train_file} 和 {target_val_file} 文件中!\033[0m")

def parse_args():
    # 创建解析器
    parser = argparse.ArgumentParser(description="将 .xml 转换为 .txt")
    
    # 添加参数
    parser.add_argument('--voc_root', type=str, default="VOCdevkit", help="PASCAL VOC路径(之后的所有路径都在voc_root下)")
    parser.add_argument('--voc_version', type=str, default="VOC2012-Lite", help="VOC 版本")
    parser.add_argument('--save_path', type=str, default="VOC2012-YOLO", help="转换后的保存目录路径")
    parser.add_argument('--train_list_name', type=str, default="train.txt", help="训练图片列表名称")
    parser.add_argument('--val_list_name', type=str, default="val.txt", help="验证图片列表名称")
    parser.add_argument('--val_size', type=float, default=0.1, help="验证集比例")
    parser.add_argument('--seed', type=int, default=42, help="随机数种子")
    parser.add_argument('--no_create_txt_for_pure_negative_sample', action='store_true', help='是否为纯负样本创建txt文件(默认创建)')
    parser.add_argument('--num_classes', type=int, default=20, help="数据集类别数(用于校验)")
    parser.add_argument('--classes', help="数据集具体类别数(用于生成 classes.json 文件)", 
                        default=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                                 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
                                 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])
    
    return parser.parse_args()


def configure_path(args):
    # 转换的训练集以及验证集对应txt文件
    args.train_txt = "train.txt"
    args.val_txt = "val.txt"

    # 转换后的文件保存目录
    args.save_file_root = os.path.join(args.voc_root, args.save_path)

    # 生成json文件
    # label标签对应json文件
    args.label_json_path = os.path.join(args.voc_root, "classes.json")
    
    # 创建一个将类别与数值关联的字典
    class_mapping = {class_name: index + 1 for index, class_name in enumerate(args.classes)}
    with open(args.label_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(class_mapping, json_file, ensure_ascii=False, indent=4)

    print(f'\033[1;31m类别列表已保存到 {args.label_json_path}\033[0m')

    # 拼接出voc的images目录，xml目录，txt目录
    args.voc_images_path = os.path.join(args.voc_root, args.voc_version, "JPEGImages")
    args.voc_xml_path = os.path.join(args.voc_root, args.voc_version, "Annotations")
    args.train_txt_path = os.path.join(args.voc_root, args.voc_version, args.train_txt)
    args.val_txt_path = os.path.join(args.voc_root, args.voc_version, args.val_txt)
    
    # 生成对应的 train.txt 和 val.txt
    generate_train_and_val_txt(args)

    # 检查文件/文件夹都是否存在
    assert os.path.exists(args.voc_images_path), f"VOC images path not exist...({args.voc_images_path})"
    assert os.path.exists(args.voc_xml_path), f"VOC xml path not exist...({args.voc_xml_path})"
    assert os.path.exists(args.train_txt_path), f"VOC train txt file not exist...({args.train_txt_path})"
    assert os.path.exists(args.val_txt_path), f"VOC val txt file not exist...({args.val_txt_path})"
    assert os.path.exists(args.label_json_path), f"label_json_path does not exist...({args.label_json_path})"
    if os.path.exists(args.save_file_root) is False:
        os.makedirs(args.save_file_root)
        print(f"创建文件夹：{args.save_file_root}")


def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args：
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def translate_info(file_names: list, save_root: str, class_dict: dict, train_val='train', args=None):
    """
    将对应xml文件信息转为yolo中使用的txt文件信息
    """
    save_txt_path = os.path.join(save_root, train_val, "labels")
    if os.path.exists(save_txt_path) is False:
        os.makedirs(save_txt_path)
    save_images_path = os.path.join(save_root, train_val, "images")
    if os.path.exists(save_images_path) is False:
        os.makedirs(save_images_path)

    for file in tqdm(file_names, desc="translate {} file...".format(train_val)):
        # 检查下图像文件是否存在（强报错！）
        img_path = os.path.join(args.voc_images_path, file + ".jpg")
        assert os.path.exists(img_path), "file:{} not exist...".format(img_path)

        # 检查xml文件是否存在（强报错！）
        xml_path = os.path.join(args.voc_xml_path, file + ".xml")
        assert os.path.exists(xml_path), "file:{} not exist...".format(xml_path)

        # 读取 xml 文件（这里修复了一下）
        # with open(xml_path) as fid:
        #     xml_str = fid.read()
        # xml = etree.fromstring(xml_str)

        with open(xml_path) as fid:
            xml_str = fid.read()
            
        # 将XML字符串编码为字节序列
        xml_bytes = xml_str.encode('utf-8')

        # 使用lxml解析字节序列的XML数据
        xml = etree.fromstring(xml_bytes)
        data = parse_xml_to_dict(xml)["annotation"]
        img_height = int(data["size"]["height"])
        img_width = int(data["size"]["width"])

        # write object info into txt
        # assert "object" in data.keys(), "file: '{}' lack of object key.".format(xml_path)
        if (not "object" in data.keys()) or (len(data["object"]) == 0):  # 没有目标，说明是纯负样本
            if args.no_create_txt_for_pure_negative_sample:  # 不为纯负样本创建txt文件
                continue
            else:  # 为纯负样本创建txt文件
                # 把纯负样本图片拷贝到指定为止
                path_copy_to = os.path.join(save_images_path, img_path.split(os.sep)[-1])
                if os.path.exists(path_copy_to) is False:
                    shutil.copyfile(img_path, path_copy_to)
                
                # 创建一个空的 .txt 文件
                with open(os.path.join(save_txt_path, file + ".txt"), "w") as f:
                    ...

                # 后面的不需要执行
                continue
                
        with open(os.path.join(save_txt_path, file + ".txt"), "w") as f:
            for index, obj in enumerate(data["object"]):
                # 获取每个object的box信息
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                class_name = obj["name"]
                
                class_index = class_dict[class_name] - 1  # 目标id从0开始

                # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
                if xmax <= xmin or ymax <= ymin:
                    print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                    continue

                # 将box信息转换到yolo格式
                xcenter = xmin + (xmax - xmin) / 2
                ycenter = ymin + (ymax - ymin) / 2
                w = xmax - xmin
                h = ymax - ymin

                # 绝对坐标转相对坐标，保存6位小数
                xcenter = round(xcenter / img_width, 6)
                ycenter = round(ycenter / img_height, 6)
                w = round(w / img_width, 6)
                h = round(h / img_height, 6)

                info = [str(i) for i in [class_index, xcenter, ycenter, w, h]]

                if index == 0:
                    f.write(" ".join(info))
                else:
                    f.write("\n" + " ".join(info))

        # copy image into save_images_path
        path_copy_to = os.path.join(save_images_path, img_path.split(os.sep)[-1])
        if os.path.exists(path_copy_to) is False:
            shutil.copyfile(img_path, path_copy_to)


def create_class_names(class_dict: dict, args):
    keys = class_dict.keys()
    with open(os.path.join(args.voc_root, "my_data_label.names"), "w") as w:
        for index, k in enumerate(keys):
            if index + 1 == len(keys):
                w.write(k)
            else:
                w.write(k + "\n")


def main(args):
    # read class_indict
    json_file = open(args.label_json_path, 'r')
    class_dict = json.load(json_file)

    # 读取train.txt中的所有行信息，删除空行
    with open(args.train_txt_path, "r") as r:
        train_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
        
    # voc信息转yolo，并将图像文件复制到相应文件夹
    translate_info(train_file_names, args.save_file_root, class_dict, "train", args=args)

    # 读取val.txt中的所有行信息，删除空行
    with open(args.val_txt_path, "r") as r:
        val_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
        
    # voc信息转yolo，并将图像文件复制到相应文件夹
    translate_info(val_file_names, args.save_file_root, class_dict, "val", args=args)

    # 创建my_data_label.names文件
    create_class_names(class_dict, args=args)


if __name__ == "__main__":
    args = parse_args()
    configure_path(args)
    
    # 美化打印 args
    print(f"\033[1;34m{args_table(args)}\033[0m")
    
    # 执行 .xml 转 .txt
    main(args)
```
</details>


我们在运行下面命令即可完成转换：

```bash
python voc2yolo.py --voc_root ./VOCdevkit \
                   --voc_version VOC2012-Lite \
                   --num_classes 20 \
                   --save_path VOC2012-YOLO
```

转换后的目录结构为：

```txt
VOCdevkit
│  classes.json
│  my_data_label.names
│  
├─VOC2012-Lite
│  │  train.txt
│  │  val.txt
│  │  
│  ├─Annotations
│  │      2007_000027.xml
│  │      2007_000032.xml
│  │      2007_000033.xml
│  │      2007_000039.xml
│  │      2007_000042.xml
│  │      2007_000061.xml
│  │      ...
│  │      
│  └─JPEGImages
│          2007_000027.jpg
│          2007_000032.jpg
│          2007_000033.jpg
│          2007_000039.jpg
│          2007_000042.jpg
│          2007_000061.jpg
│          ...
│
└─VOC2012-YOLO
    ├─train
    │  ├─images
    │  │      2007_000032.jpg
    │  │      2007_000033.jpg
    │  │      2007_000039.jpg
    │  │      2007_000042.jpg
    │  │      2007_000061.jpg
    │  │      ...
    │  │
    │  └─labels
    │          2007_000032.txt
    │          2007_000033.txt
    │          2007_000039.txt
    │          2007_000042.txt
    │          2007_000061.txt
    │          ...
    │
    └─val
        ├─images
        │      2007_000027.jpg
        │      ...
        │
        └─labels
                2007_000027.txt
                ...
```

<div align=center>
    <img src=./imgs_markdown/2023-10-18-21-14-28.png
    width=50%>
</div>

## 1.6 YOLOv5 配置文件变动

根据 `.yaml` 配置文件变动而变动的，这里我们复制 `coco128.yaml` 为 `custom_dataset.yaml` 为例:

```yaml
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
# COCO128 dataset https://www.kaggle.com/ultralytics/coco128 (first 128 images from COCO train2017) by Ultralytics
# Example usage: python train.py --data coco128.yaml
# parent
# ├── yolov5
# └── datasets
#     └── coco128  ← downloads here (7 MB)


# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: VOCdevkit/VOC2012-YOLO  # dataset root dir
train: train/images  # train images (relative to 'path') 128 images
val: val/images  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  ...


# Download script/URL (optional)
download: https://ultralytics.com/assets/coco128.zip
```

此时我们就可以使用这个数据集进行 YOLOv5 的模型训练了！

## 1.7 【补充】如果标签格式为 .json

### 1.7.1 将负样本放在正样本所属文件夹下

**说明**：我们应该把正负样本放在同一个文件夹下，如 `JPEGImages`，这样我们再为没有标签文件的负样本生成 .json 文件。

> 单独为负样本生成 .json 文件，之后再合并也是可以的。

```python
"""
    描述：
        1. 检查负样本数量是否正确；
        2. 检查正样本数量是否正确；
        3. 检查Annotations数量是否正确
"""
import os
import shutil
import tqdm


"""============================ 需要修改的地方 ==================================="""
# 数据所在路径
BASE_PATH = 'EXAMPLE_DATASET/DATASET_A'
CHECK_NUM = False  # 是否检查样本数量
POS_SAMPLE_NUM = 6914  # 正样本数量 -> 6914
NEG_SAMPLE_NUM = 515  # 负样本数量 -> 515
"""==============================================================================="""

# 组合路径
source_path = os.path.join(BASE_PATH, "VOC2007")  # EXAMPLE_DATASET/VOC2007
pos_image_path = os.path.join(source_path, "JPEGImages")  # EXAMPLE_DATASET/VOC2007/JPEGImages
annotation_path = os.path.join(source_path, "Annotations")  # EXAMPLE_DATASET/VOC2007/Annotations
neg_image_path = os.path.join(source_path, "neg_samples")  # EXAMPLE_DATASET/VOC2007/neg_samples

# 获取所有图片和标签
pos_image_list = os.listdir(pos_image_path)
annotation_list = os.listdir(annotation_path)
neg_image_list = os.listdir(neg_image_path)

# 过滤只包括特定类型的图像文件（这里是.jpg和.png）
pos_image_list = [file for file in pos_image_list if file.lower().endswith(('.jpg', '.png'))]
annotation_list = [file for file in annotation_list if file.lower().endswith(('.json', '.xml'))]
neg_image_list = [file for file in neg_image_list if file.lower().endswith(('.jpg', '.png'))]

# 记录实际数据数量
POS_IMG_NUM = len(pos_image_list)
ANNOTATIONS_NUM = len(annotation_list)
NEG_IMG_NUM = len(neg_image_list)

# 检查数据是否正确
if CHECK_NUM:
    assert POS_SAMPLE_NUM == POS_IMG_NUM, f"\033[1;31m正样本数量({POS_SAMPLE_NUM})和实际正样本数量({POS_IMG_NUM})不一致！\033[0m"
    assert CHECK_NUM and POS_IMG_NUM == ANNOTATIONS_NUM, f"\033[1;31m实际正样本数量({POS_IMG_NUM})和实际标签数量({ANNOTATIONS_NUM})不一致！\033[0m"
    assert CHECK_NUM and NEG_SAMPLE_NUM == NEG_IMG_NUM, f"\033[1;31m负样本数量({NEG_SAMPLE_NUM})和实际负样本数量({NEG_IMG_NUM})不一致！\033[0m"
else:
    print("\033[1;31m💡请注意：跳过了数据检查！\033[0m")

SKIP_NUM = 0
SUCCEED_NUM = 0

# 创建进度条
progress_bar = tqdm.tqdm(total=NEG_IMG_NUM, desc="Copy neg2pos", unit=" img")
for neg_image_name in neg_image_list:
    # 分离文件名和后缀
    image_pre, image_ext = os.path.splitext(neg_image_name)

    # 确定图片的路径 -> EXAMPLE_DATASET/VOC2007/neg_samples/xxxx_yyyy_xxxx_yyyy.jpg
    src_img_path = os.path.join(neg_image_path, neg_image_name)
    # 确定保存的路径 -> EXAMPLE_DATASET/VOC2007/JPEGImages/xxxx_yyyy_xxxx_yyyy.jpg
    target_img_path = os.path.join(pos_image_path, neg_image_name)

    # 判断对应的json文件是否存在
    if os.path.exists(target_img_path):
        SKIP_NUM += 1
        progress_bar.update(1)
        continue
    
    # 开始复制
    shutil.copy(src=src_img_path, dst=target_img_path)
    SUCCEED_NUM += 1
    progress_bar.update(1)

print(f"SUCCEED NUM: {SUCCEED_NUM}/{NEG_IMG_NUM}")
print(f"SKIP NUM: {SKIP_NUM}/{NEG_IMG_NUM}")

if SUCCEED_NUM + SKIP_NUM == NEG_SAMPLE_NUM:
    print("\n\033[1;36mNo Problems in Copying\033[0m\n")
    # 再次检查数据数量
    if POS_SAMPLE_NUM + NEG_SAMPLE_NUM == POS_IMG_NUM + SUCCEED_NUM:
        print(f"\n\033[1;36m👌预想正负样本数量({POS_SAMPLE_NUM} + {NEG_SAMPLE_NUM}) == 实际的正负样本数量({POS_IMG_NUM} + {SUCCEED_NUM})\033[0m\n")
    else:
        print(f"\n\033[1;31m🤡出现了问题：预想正负样本数量({POS_SAMPLE_NUM} + {NEG_SAMPLE_NUM}) != 实际的正负样本数量({POS_IMG_NUM} + {SUCCEED_NUM})\033[0m\n")
else:
    print(f"\n\033[1;31m🤡有问题: 成功/负样本数量 -> {SUCCEED_NUM}/{NEG_SAMPLE_NUM}\033[0m\n")
```

### 1.7.2 为负样本生成空的 .json 文件

没啥好说的，直接生成就行了。

```python
"""
    描述：为所有图片创建空的json文件（如果json文件存在则跳过）
    作用：为负样本生成对应的json文件
"""

import numpy as np
import os
import cv2
import json
import tqdm


"""============================ 需要修改的地方 ==================================="""
# 图片所在文件夹路径
source_folder_path = 'EXAMPLE_DATASET/VOC2007/JPEGImages'

# json文件路径
target_folder_path = 'EXAMPLE_DATASET/VOC2007/Annotations'

# 负样本数量
NEG_SAMPLE_NUM = 1024
"""==============================================================================="""

# 获取所有图片
image_list = os.listdir(source_folder_path)
# 过滤只包括特定类型的图像文件（这里是.jpg和.png）
image_list = [file for file in image_list if file.lower().endswith(('.jpg', '.png'))]
TOTAL_NUM = len(image_list)
SKIP_NUM = 0
SUCCEED_NUM = 0

# 创建进度条
progress_bar = tqdm.tqdm(total=len(image_list), desc="json2yolo", unit=" .json")
for image_name in image_list:
    # 分离文件名和后缀
    image_pre, image_ext = os.path.splitext(image_name)

    # 确定保存的路径
    target_path = os.path.join(target_folder_path, image_pre) + '.json'
    # 确定图片的路径
    img_file = os.path.join(source_folder_path, image_name)

    # 判断对应的json文件是否存在
    if os.path.exists(target_path):
        SKIP_NUM += 1
        progress_bar.update(1)
        continue

    img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_COLOR)
    height, width, _ = img.shape
    content = {"version": "0.2.2",
               "flags": {},
               "shapes": [],
               "imagePath": "{}.jpg".format(image_pre),
               "imageData": None,
               "imageHeight": height,
               "imageWidth": width
               }
    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)

    with open(target_path, 'w') as f:
        json.dump(content, f, indent=2)
    SUCCEED_NUM += 1
    progress_bar.update(1)

print(f"SUCCEED NUM: {SUCCEED_NUM}/{TOTAL_NUM}")
print(f"SKIP NUM: {SKIP_NUM}/{TOTAL_NUM}")

if SUCCEED_NUM == NEG_SAMPLE_NUM:
    print("\n\033[1;36m👌No Problems\033[0m\n")
else:
    print(f"\n\033[1;31m🤡有问题: 成功/负样本数量 -> {SUCCEED_NUM}/{NEG_SAMPLE_NUM}\033[0m\n")
```

### 1.7.3 json 转 yolo 的 txt

```python
"""
    json转yolo的txt
"""

import os
import cv2
import json
import numpy as np
import tqdm

"""============================ 需要修改的地方 ==================================="""
# 标签字典
label_dict = {'cls_1': 0,
              'cls_2': 1,
              }
# 文件夹路径
base_path = 'EXAMPLE_DATASET/VOC2007'

OVERRIDE = False  # 是否要覆盖已存在txt文件
use_kpt_check = False
"""==============================================================================="""

path = os.path.join(base_path, 'Annotations')
all_json_list = os.listdir(path)
TOTAL_NUM = len(all_json_list)
SUCCESSES_NUM = 0
SKIP_NUM = 0
ERROR_NUM = 0
ERROR_LIST = []

# 创建进度条
progress_bar = tqdm.tqdm(total=len(all_json_list), desc="json2yolo", unit=" .txt")

for idx, anno_name in enumerate(all_json_list):  # anno_json = 'xxxxxx_yyyyyyy_ccccc.json'
    target_path = os.path.join(base_path, 'labels', anno_name.replace('.json', '.txt'))
    if not OVERRIDE and os.path.exists(target_path):
        SKIP_NUM += 1
        continue

    progress_bar.set_description(f"\033[1;31m{anno_name}\033[0m")

    with open(os.path.join(path, anno_name), 'r') as fr:
        result = json.load(fr)

    img = cv2.imread(os.path.join(base_path, 'JPEGImages',
                     anno_name).replace('.json', '.jpg'))
    h_, w_ = img.shape[0:2]
    object_info = result['shapes']

    # exist_ok=True 表示如果目标目录已存在，则不会引发异常，而是默默地跳过创建该目录的步骤
    os.makedirs(os.path.join(base_path, 'labels'), exist_ok=True)
    with open(target_path, 'w') as target_file:
        try:
            for line in object_info:
                label = label_dict[line['label']]
                # label = 0 if line['label'] == 'chepai' else 1
                kpt = np.array(line['points'])
                if use_kpt_check and (kpt[1][0] > kpt[3][0] and kpt[1][1] > kpt[3][1]):
                    continue
                else:
                    x1, y1, x2, y2 = kpt[0][0], kpt[0][1], kpt[1][0], kpt[1][1]
                    xc, yc, w, h = x1 + (x2-x1)/2, y1 + (y2-y1)/2, x2-x1, y2-y1

                    line = '{} {} {} {} {}'.format(
                        label, xc/w_, yc/h_, w/w_, h/h_)
                    target_file.write(line+'\n')
            SUCCESSES_NUM += 1

        except:
            ERROR_NUM += 1
            ERROR_LIST.append(os.path.join(path, anno_name))

    progress_bar.update(1)
progress_bar.close()

for _ef in ERROR_LIST:
    print(_ef)

print(f"json2yolo已完成，详情如下：\n\t"
      f"👌成功: {SUCCESSES_NUM}/{TOTAL_NUM}\n\t"
      f"👌跳过: {SKIP_NUM}/{TOTAL_NUM}\n\t"
      f"🤡失败: {ERROR_NUM}/{TOTAL_NUM}")
```

### 1.7.4 划分数据，并生成数据集

```python
"""
    生成数据集
"""
# 导入所需库
import os
from sklearn.model_selection import train_test_split
import shutil
import tqdm


"""============================ 需要修改的地方 ==================================="""
test_size = 0.01
OVERRIDE = False

# 图片文件夹路径
target_image_folder = "EXAMPLE_DATASET/VOC2007/JPEGImages"

# txt文件夹路径
target_label_folder = "EXAMPLE_DATASET/VOC2007/labels"

# 输入文件夹路径
output_folder = "EXAMPLE_DATASET"
"""==============================================================================="""

# 读取所有.txt文件
labels = [label for label in os.listdir(target_label_folder) if label.endswith(".txt")]

TOTAL_NUM = len(labels)

print(f"预计验证集样本数量为: \033[1;31m{round(TOTAL_NUM * test_size)}\033[0m，请输入 \033[1;31myes\033[0m 继续 | 输入其他退出")

_INPUT = input()
if _INPUT != "yes":
    exit()

# 使用sklearn进行数据集划分
train_list, val_list = train_test_split(labels, test_size=test_size, random_state=42)
print(f"训练集大小: {len(train_list)}/{TOTAL_NUM} | 验证集大小: {len(val_list)}/{TOTAL_NUM}")

# 定义保存训练集和验证集的文件夹路径
train_image_folder = os.path.join(output_folder, "train", "images")
train_label_folder = os.path.join(output_folder, "train", "labels")
val_image_folder = os.path.join(output_folder, "val", "images")
val_label_folder = os.path.join(output_folder, "val", "labels")
print(f"train_image_folder: {train_image_folder}")
print(f"train_label_folder: {train_label_folder}")
print(f"val_image_folder: {val_image_folder}")
print(f"val_label_folder: {val_label_folder}")

# 创建保存文件夹
os.makedirs(train_image_folder, exist_ok=True)
os.makedirs(train_label_folder, exist_ok=True)
os.makedirs(val_image_folder, exist_ok=True)
os.makedirs(val_label_folder, exist_ok=True)

print("=" * 50)

# 将训练集的图片和标签拷贝到对应文件夹
progress_bar = tqdm.tqdm(total=len(train_list), desc="Copying in \033[1;31mtrain\033[0m", unit=" file")
TRAIN_SUCCESSES_NUM = 0
TRAIN_SKIP_NUM = 0
for label in train_list:
    label_path = os.path.join(target_label_folder, label)
    image_path = os.path.join(target_image_folder, label.replace(".txt", ".jpg"))
    
    # 定义目标路径
    target_img = os.path.join(train_image_folder, label.replace(".txt", ".jpg"))
    target_label = os.path.join(train_label_folder, label)
    if not OVERRIDE and os.path.exists(target_img) and target_label:
        TRAIN_SKIP_NUM += 1
        progress_bar.update(1)
        continue

    shutil.copy(image_path, target_img)
    shutil.copy(label_path, target_label)
    TRAIN_SUCCESSES_NUM += 1
    progress_bar.update(1)
progress_bar.close()

# 将验证集的图片和标签拷贝到对应文件夹
progress_bar = tqdm.tqdm(total=len(train_list), desc="Copying in \033[1;31mvalidation\033[0m", unit=" file")
VAL_SUCCESSES_NUM = 0
VAL_SKIP_NUM = 0
for label in val_list:
    label_path = os.path.join(target_label_folder, label)
    image_path = os.path.join(target_image_folder, label.replace(".txt", ".jpg"))

    # 定义目标路径
    target_img = os.path.join(val_image_folder, label.replace(".txt", ".jpg"))
    target_label = os.path.join(val_label_folder, label)
    
    if not OVERRIDE and os.path.exists(target_img) and target_label:
        VAL_SKIP_NUM += 1
        progress_bar.update(1)
        continue

    shutil.copy(image_path, target_img)
    shutil.copy(label_path, target_label)
    VAL_SUCCESSES_NUM += 1
    progress_bar.update(1)
progress_bar.close()

print(
    f"\n数据集创建完毕，详情如下：\n\t"
    f"训练集:\n\t\t"
    f"图片路径: {train_image_folder}\n\t\t"
    f"标签路径: {train_label_folder}\n\t\t\t"
    f"👌成功: {TRAIN_SUCCESSES_NUM}/{len(train_list)}\n\t\t\t"
    f"👌跳过: {TRAIN_SKIP_NUM}/{len(train_list)}\n\t"
    
    f"验证集:\n\t\t"
    f"图片路径: {val_image_folder}\n\t\t"
    f"标签路径: {val_label_folder}\n\t\t\t"
    f"👌成功: {VAL_SUCCESSES_NUM}/{len(val_list)}\n\t\t\t"
    f"👌跳过: {VAL_SKIP_NUM}/{len(val_list)}"
)
```

## 1.8 【补充】随机挑选数据组成测试集

如果我们有一批模型从来没有见过的（差异非常大）的数据，那么我们可以随机挑选数据组成测试集，从而快速测试。

```python
import os
import tqdm
import random
import shutil
import subprocess


"""============================ 需要修改的地方 ==================================="""
# 源视频路径
src_folder = 'Addition_dataset'

# 保存的路径
dst_folder_origin = 'data-test'

TEST_IMG_NUM = 100  # 测试图片数量
record_time = "20231114"  # 时间
other_content = ""  # 其他备注
"""==============================================================================="""

# 读取目标文件夹中的图片
imgs_list = os.listdir(src_folder)
# 过滤只包括特定类型的图像文件（这里是.jpg和.png）
imgs_list = [file for file in imgs_list if file.lower().endswith(('.jpg', '.png'))]

# 随机数组
random.shuffle(imgs_list)  # in-place操作

# 组成路径并创建文件夹
if other_content:
    dst_folder = dst_folder_origin + f"-{record_time}-{other_content}"
else:    
    dst_folder = dst_folder_origin + f"-{record_time}"
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder, exist_ok=True)

# 创建一个tqdm进度条对象
progress_bar = tqdm.tqdm(total=TEST_IMG_NUM, desc="随机抽取图片组成测试集", unit="img")
for count, img_name in enumerate(imgs_list):
    if count >= TEST_IMG_NUM:
        break
    progress_bar.set_description(f"selected \033[1;31m{img_name}\033[0m")
    
    # 确定路径
    src_path = os.path.join(src_folder, img_name)
    dst_path = os.path.join(dst_folder, img_name)
    
    # 开始复制
    shutil.copy(src=src_path, dst=dst_path)
    
    progress_bar.update(1)
progress_bar.close()

# 压缩文件夹
# 切换当前工作目录到源文件夹所在的位置
os.chdir(dst_folder_origin)

if other_content:
    zip_file_name = f"{record_time}-{other_content}.7z"
else:
    zip_file_name = f"{record_time}.7z"
zip_command = f"7z a {zip_file_name} {dst_folder.split('/')[-1]}/*"

subprocess.run(zip_command, shell=True)

print(f"复制完成，一共获得了 {TEST_IMG_NUM} 张测试图片，路径为: {dst_folder}")
print(f"压缩完成，压缩包路径为: {os.path.join(dst_folder_origin, zip_file_name)}")
```

在 Linux 中，如果最后的压缩程序没有运行，请安装 `7zip`：

```bash
sudo apt install p7zip-full
```

## 1.9 【补充】如果数据集有好几部分 | 合并多个训练文件夹

有时候我们的数据集是由好几部分组成的，比如：
1. `DATASET_PART_A`
2. `DATASET_PART_B`
3. `DATASET_PART_C`

<kbd>Q</kbd>：那么我们需要把它们合在一起组成 `DATASET_PART_FULL` 吗？
<kbd>A</kbd>：说实话，我之前一直是这样做的，那是我发现这样是非常蠢的 —— 数据集耦合性拉满，而且原来的碎片也不能丢掉（当做备份）。在 YOLOv5 中，其实是支持多个文件夹的，具体如下：

```yaml
path: ../datasets/coco
train: train2017.txt
val: val2017.txt
test: test-dev2017.txt

# Classes
names:
  0: person
  1: bicycle
  2: car
  ...
```

上面是 `coco.yaml` 文件的内容，这里我们假设我们的数据也保存在 `../datasets/coco` 中，但有 3 个子文件夹：

1. `../datasets/coco/partA`
2. `../datasets/coco/partB`
3. `../datasets/coco/partC`

此时我们可以将 yaml 文件改为如下所示的：

```yaml
path: ../datasets/coco
train: 
  - partA/train2017.txt
  - partB/train2017.txt
  - partC/train2017.txt
val:
  - partA/val2017.txt
  - partB/val2017.txt
  - partC/val2017.txt
test:
  - partA/test-dev2017.txt
  - partB/test-dev2017.txt
  - partC/test-dev2017.txt

# Classes
names:
  0: person
  1: bicycle
  2: car
  ...
```

这样 YOLOv5 在加载数据集的时候会将三部分的数据都加载上。三个不同的数据集也更加方便管理。

**注意**：YOLOv5 默认会为数据集保留一个 `.cache` 文件，以便下次快速加载数据集，由于我们的数据集分为三个部分，因此 `.cache` 只会保存在第一个文件夹中，即 `partA` 文件夹下。

# 2. 模型选择

我们需要选择一个合适的模型来进行训练，在这里，我们选择 YOLOv5s，这是第二小和速度最快的可用模型。

<div align=center>
    <img src=./imgs_markdown/2023-10-18-21-18-35.png
    width=100%>
</div>

# 3. 模型训练

通过指定数据集、批次大小、图像大小以及预训练权重 `--weights yolov5s.pt`在我们自建的数据集上训练 YOLOv5s 模型。

```bash
export CUDA_VISIBLE_DEVICES=4
python train.py --img 640 \
                --epochs 150 \
                --data custom_dataset.yaml \
                --weights weights/yolov5s.pt \
                --batch-size 32 \
                --single-cls \
                --project runs/train \
                --cos-lr
```



为了加快训练速度，可以添加 `--cache ram` 或 `--cache disk` 选项（需要大量的内存/磁盘资源）。所有训练结果都会保存在 `runs/train/` 目录下，每次训练都会创建一个递增的运行目录，例如 `runs/train/exp2`、`runs/train/exp3` 等等。

## 2.5 可视化

训练结果会自动记录在 Tensorboard 和 CSV 日志记录器中，保存在 `runs/train` 目录下，每次新的训练都会创建一个新的实验目录，例如 `runs/train/exp2`、`runs/train/exp3` 等。

该目录包含了训练和验证的统计数据、马赛克图像、标签、预测结果、以及经过增强的马赛克图像，还包括 Precision-Recall（PR）曲线和混淆矩阵等度量和图表。

<div align=center>
    <img src=./imgs_markdown/2023-10-18-21-25-37.png
    width=100%>
</div>

结果文件 `results.csv` 在每个 Epoch 后更新，然后在训练完成后绘制为 `results.png`（如下所示）。我们也可以手动绘制任何 `results.csv` 文件：

```python
from utils.plots import plot_results

plot_results('path/to/results.csv')  # plot 'results.csv' as 'results.png'
```

<div align=center>
    <img src=./imgs_markdown/2023-10-18-21-25-55.png
    width=100%>
</div>

# 4. <kbd>补充</kbd> 现实场景中数据集构建遇到的问题

## 4.1 数据集中图片和标注数量不一致怎么办？

有时候我们标注完所有的图片后，会手动检查一遍，删除掉一些不合理的图片。删除图片我推荐使用 Windows 自带的图片软件，如下图所示：

<div align=center>
    <img src=./imgs_markdown/2023-10-24-15-41-02.png
    width=50%>
</div>

之后我们打开图片，使用 <kbd>←</kbd><kbd>→</kbd> 方向键即可浏览上一张图片和下一张图片。对于不合理的图片，我们可以直接使用键盘快捷键 <kbd>Delete</kbd> 来删除此时显示的图片。

在删除完所有不合理的图片后，我们会发现，此时图片数量和标注文件数量不一致了，需要进行处理，这里我推荐使用下面的脚本：

> <kbd>Note</kbd>：<font color='red'>在运行下面脚本的时候一定要备份数据集！</font>

```python
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
```

上面的脚本可以检查图片和标注文件，看标注文件是否有对应的图片，如果没有，则删除标注文件。

## 4.2 数据集中有纯负样本怎么办？

在实际任务中，我们难免会有一张图片是负样本的情况，此时这张图片是没有任何 Object 的。我们一般使用 LabelImg 来标注图片，但 LabelImg 不会对没有 Object 的图片生成对应的 `.xml` 文件，此时我们运行上面给的 `voc2yolo.py` 文件就会报错，因为我们断言了 `.xml` 是否存在。那么我们直接 `continue` 可以吗？其实是可以的，但是我们一般是想往数据集中添加一定的纯负样本的，直接 `continue` 就没有办法添加纯负样本了，那我们该怎么办？

其实方法也比较简单，首先为所有的图片生成一个 `.xml` 文件，脚本如下：

```python
import os
import xml.dom.minidom
from tqdm import tqdm


# 为哪些图片生成 .xml 文件？
img_path = '/mnt/c/Users/Le0v1n/Desktop/测试案例/Datasets/exp_1/JPEGImages'

# 将生成的 .xml 文件保存到哪个文件夹下？
xml_path = '/mnt/c/Users/Le0v1n/Desktop/测试案例/Datasets/exp_1/Empty_Annotations'

# 获取图像文件列表
img_files = os.listdir(img_path)

# 使用tqdm创建进度条
for img_file in tqdm(img_files, desc="生成XML文件"):
    img_name = os.path.splitext(img_file)[0]

    # 创建一个空的DOM文档对象
    doc = xml.dom.minidom.Document()
    # 创建名为annotation的根节点
    annotation = doc.createElement('annotation')
    # 将根节点添加到DOM文档对象
    doc.appendChild(annotation)

    # 添加folder子节点
    folder = doc.createElement('folder')
    folder_text = doc.createTextNode('VOC2007')
    folder.appendChild(folder_text)
    annotation.appendChild(folder)

    # 添加filename子节点
    filename = doc.createElement('filename')
    filename_text = doc.createTextNode(img_file)
    filename.appendChild(filename_text)
    annotation.appendChild(filename)

    # 添加path子节点
    path = doc.createElement('path')
    path_text = doc.createTextNode(img_path + '/' + img_file)  # 修正路径
    path.appendChild(path_text)
    annotation.appendChild(path)

    # 添加source子节点
    source = doc.createElement('source')
    database = doc.createElement('database')
    database_text = doc.createTextNode('Unknown')
    source.appendChild(database)
    database.appendChild(database_text)
    annotation.appendChild(source)

    # 添加size子节点
    size = doc.createElement('size')
    width = doc.createElement('width')
    width_text = doc.createTextNode('1280')
    height = doc.createElement('height')
    height_text = doc.createTextNode('720')
    depth = doc.createElement('depth')
    depth_text = doc.createTextNode('3')
    size.appendChild(width)
    width.appendChild(width_text)
    size.appendChild(height)
    height.appendChild(height_text)
    size.appendChild(depth)
    depth.appendChild(depth_text)
    annotation.appendChild(size)

    # 添加segmented子节点
    segmented = doc.createElement('segmented')
    segmented_text = doc.createTextNode('0')
    segmented.appendChild(segmented_text)
    annotation.appendChild(segmented)

    # 将XML写入文件
    xml_file_path = os.path.join(xml_path, f'{img_name}.xml')
    with open(xml_file_path, 'w+', encoding='utf-8') as fp:
        doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding='utf-8')
```

**注意路径**：
1. `img_path`: 对哪个文件夹下的图片生成 .xml 文件
2. `xml_path`: 将生成的 .xml 文件放在哪个文件夹里

> 有些同学可能会担心，在 `voc2yolo.py` 中会通过图片的尺寸进行坐标转换，但是你要记住，那是对于有 Object 的图片而言的，对于纯负样本而言，没有任何 Object，也就不会进行坐标转换，所以这里随便写了一个 1280×720 是合理的。

接下来，我们需要将之前标注好的 `.xml` 文件（是自己标注的，不是生成的文件），复制一下，然后粘贴到生成的 `.xml` 文件中。当系统提示有重名文件时，全部覆盖即可。这样，所有的图片都会有自己的 `.xml` 文件了。


此时，我们再运行 `voc2yolo.py` 文件，它会对纯负样本生成一个 `.txt` 文件。

> <kbd>Note</kbd>：在 `voc2yolo.py` 脚本中，有一个名为 `--no_create_txt_for_pure_negative_sample` 的参数。当该参数被触发时，脚本不会为纯负样本生成 `.txt` 文件（默认会生成 `.txt` 文件）

# 知识来源

1. [Ultralytics YOLOv5 Docs](https://docs.ultralytics.com/yolov5/tutorials)
2. [【CSDN】PASCAL VOC 2012 数据集讲解与制作自己的数据集](https://blog.csdn.net/weixin_44878336/article/details/124540069)
3. [【Bilibili】PASCAL VOC 2012 数据集讲解与制作自己的数据集](https://www.bilibili.com/video/BV1kV411k7D8)
4. [trans_voc2yolo.py](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_object_detection/yolov3_spp/trans_voc2yolo.py)