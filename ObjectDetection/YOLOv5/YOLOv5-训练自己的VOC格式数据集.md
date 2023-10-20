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
    :param file_names:
    :param save_root:
    :param class_dict:
    :param train_val:
    :return:
    """
    save_txt_path = os.path.join(save_root, train_val, "labels")
    if os.path.exists(save_txt_path) is False:
        os.makedirs(save_txt_path)
    save_images_path = os.path.join(save_root, train_val, "images")
    if os.path.exists(save_images_path) is False:
        os.makedirs(save_images_path)

    for file in tqdm(file_names, desc="translate {} file...".format(train_val)):
        # 检查下图像文件是否存在
        img_path = os.path.join(args.voc_images_path, file + ".jpg")
        assert os.path.exists(img_path), "file:{} not exist...".format(img_path)

        # 检查xml文件是否存在
        xml_path = os.path.join(args.voc_xml_path, file + ".xml")
        assert os.path.exists(xml_path), "file:{} not exist...".format(xml_path)

        # read xml
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)["annotation"]
        img_height = int(data["size"]["height"])
        img_width = int(data["size"]["width"])

        # write object info into txt
        assert "object" in data.keys(), "file: '{}' lack of object key.".format(xml_path)
        if len(data["object"]) == 0:
            # 如果xml文件中没有目标就直接忽略该样本
            print("Warning: in '{}' xml, there are no objects.".format(xml_path))
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

我们在运行下面命令即可完成转换：

```bash
python voc2yolo.py --voc_root ./VOCdevkit --voc_version VOC2012-Lite
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

# 知识来源

1. [Ultralytics YOLOv5 Docs](https://docs.ultralytics.com/yolov5/tutorials)
2. [【CSDN】PASCAL VOC 2012 数据集讲解与制作自己的数据集](https://blog.csdn.net/weixin_44878336/article/details/124540069)
3. [【Bilibili】PASCAL VOC 2012 数据集讲解与制作自己的数据集](https://www.bilibili.com/video/BV1kV411k7D8)
4. [trans_voc2yolo.py](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_object_detection/yolov3_spp/trans_voc2yolo.py)