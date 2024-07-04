# 0. 前言

## 0.1 YOLO-World介绍

YOLO-World 是一种基于 YOLO系列目标检测器的创新方法，它通过视觉语言建模和大规模数据集的预训练，增强了 YOLO 的开放词汇检测能力。具体来说，YOLO-World 引入了一种新的可重参数化的视觉-语言路径聚合网络（RepVL-PAN）和区域-文本对比损失，以促进视觉和语言信息之间的交互。这种方法在零样本（zero-shot）情况下高效地检测各种物体，并且在具有挑战性的 LVIS 数据集上表现出色，实现了高准确率和高速度的检测。

YOLO-World 的核心创新点包括：
- 实时解决方案：利用 CNN 的计算速度，提供快速的开放词汇检测解决方案。
- 效率和性能：在不牺牲性能的前提下降低计算和资源需求，支持实时应用。
- 利用离线词汇进行推理：引入了 "先提示后检测" 的策略，使用预先计算的自定义提示来提高效率。
- 卓越的基准测试：在标准基准测试中，YOLO-World 的速度和效率超过了现有的开放词汇检测器。
- 应用广泛：YOLO-World 的创新方法为众多视觉任务带来了新的可能性。

## 0.2 前置知识

前置知识包括：

| 名称         | 文内链接         |
| :----------- | :--------------- |
| 零样本       | [文内链接](#6.1) |
| CLIP         | [文内链接](#6.2) |
| 开集目标检测 | [文内链接](#6.3) |
| LVIS数据集   | [文内链接](#6.4) |

请点击对应链接跳转到本文的对应位置。

# 1. 安装

目前YOLO-World有两个仓库：

1. [官方基于MMOpenLab实现](https://github.com/AILab-CVC/YOLO-World?tab=readme-ov-file)
2. [Ultralytics实现](https://github.com/ultralytics/ultralytics)

官方实现相比Ultralytics实现有更多的细节，因此这里我们使用官方基于MMOpenLab实现，具体安装请见[installation.md](https://github.com/AILab-CVC/YOLO-World/blob/master/docs/installation.md)。下面是我自己的安装过程：

```bash
# 安装虚拟环境
conda create -n yolo-world python=3.9
conda activate yolo-world

# 根据cuda版本安装PyTorch（如果安装慢，则可以在后面添加 -i https://pypi.tuna.tsinghua.edu.cn/simple）
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install requests==2.28.2 tqdm==4.65.0 rich==13.4.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U openmim
mim install mmcv=2.1.0
mim install mmdet=3.3.0
mim install mmcv=2.1.0
mim install mmcv-lite=2.0.1
mim install mmengine=0.10.4
mim install mmyolo=0.6.0
```

<font color='red'><b>如果MMYOLO安装失败</b></font>（安装成功了则不需要了），那么从MMYOLO官方仓库下载项目压缩包，之后再：

```bash
# 解压
7z x mmyolo-main.zip -o.

# 把文件夹名字从mmyolo-main修改为mmyolo
mv mmyolo-main mmyolo

# 安装mmyolo
pip install -e mmyolo
```

安装完成后再安装其他依赖（其他MMOpenlab的库如果安装失败，那么也可以使用这样的方式来进行）：

```bash
# 安装其他依赖包
pip install opencv-python --upgrade
pip install opencv-python-headless --upgrade
pip install timm==0.6.13 transformers==4.36.2 albumentations==1.4.4
pip install gradio==4.16.0 supervision
pip install onnx onnxruntime onnxsim
```

在安装YOLO-World项目之前，需要打开`pyproject.toml`文件，将`dependencies`修改为如下内容：

```toml
dependencies = [
    "wheel",
    "torch==2.1.2",
    "torchvision==0.16.2",
    "transformers",
    "tokenizers",
    "numpy",
    "opencv-python",
    "supervision==0.19.0",
    "openmim",
    "mmcv-lite==2.0.1",
    "mmdet==3.3.0",
    "mmengine==0.10.4",
    "mmcv==2.1.0",
#    'mmyolo @ git+https://github.com/onuralpszr/mmyolo.git',

]
```

<kbd>Ctrl + S</kbd>保存后，安装YOLO-World项目：

```bash
# 安装yolo-world项目
pip install -e .
```

检查`third_party`文件夹是否为空，如果为空，那么将`mm-yolo`这个文件夹放到`third_party`中。

# 2. 数据准备 <a id=Title_2></a>

## 2.1 概况

YOLO-World的预训练模型采用了下表列出的几个数据集：

| Data         | Samples |    Type    | Boxes  | Description                                                         |
| :----------- | :-----: | :--------: | :----: | :------------------------------------------------------------------ |
| Objects365v1 |  609k   | detection  | 9,621k | 一个大规模的对象检测数据集，包含超过60万张图像和近1千万个边界框     |
| GQA          |  621k   | grounding  | 3,681k | 包含超过62万张图像和超过368万对问答对的数据集，用于视觉问答任务     |
| Flickr       |  149k   | grounding  |  641k  | 一个包含约14万张图像和641k问答对的数据集，用于视觉问答任务          |
| CC3M-Lite    |  245k   | image-text |  821k  | 一个包含24.5万图像-标题对的数据集，专注于跨模态匹配，共有821k个实例 |

其中：
- **detection**：指的是对象检测任务，算法需要识别图像中的对象并为它们绘制边界框。
- **grounding**：将自然语言描述与图像中的具体物体建立联系的过程。
- **image-text**：涉及将图像内容与相应的文本描述进行匹配的任务，可能包括图像标注、图像描述生成等。

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：grounding和image-text有什么区别？
🥳 𝑨𝒏𝒔𝒘𝒆𝒓：在视觉领域，"grounding"和"image-text"是两个相关但有所区别的概念：

1. **Grounding**：
   - 在视觉接地（Visual Grounding）任务中，"grounding"指的是将文本描述中的词汇或短语与图像中的具体物体或场景相匹配的过程。这通常涉及到理解和关联语言描述与视觉信息，以识别图像中与文本描述相对应的物体或区域。
   - Grounding任务可以视为一种跨模态的映射，<font color='blue'><b>它要求模型不仅要理解文本的含义，还要将这些文本与图像中的具体视觉实体关联起来</b></font>。

2. **Image-Text**：
   - "Image-Text"通常指的是图像和文本对，这种数据对可以用于多种任务，例如图像描述生成、视觉问答、图像检索等。在这些任务中，图像和文本并不是直接相互映射，而是作为一种多模态数据存在，用于训练和评估模型对视觉和语言信息的联合理解。
   - Image-Text任务更侧重于图像和文本之间的语义关联，可能<font color='blue'><b>不要求模型在图像中精确地定位与文本描述直接对应的物体或区域，而是更关注整体的语义一致性</b></font>。

总的来说，"grounding"更侧重于文本描述与图像中具体物体的精确匹配和定位，而"image-text"则是更广泛的概念，涵盖了图像和文本之间的各种语义关联任务，不一定要求精确的物体定位。

> 在YOLO-World这样的模型中，"grounding"能力使得模型能够根据文本描述检测图像中的物体，而"image-text"数据则可能用于模型的预训练，以提高对视觉和语言信息的联合理解能力。

## 2.2 数据集目录结构

YOLO-World项目的数据集都放入 `data` 目录中，如：

```
├── coco
│   ├── annotations
│   ├── lvis
│   ├── train2017
│   ├── val2017
├── flickr
│   ├── annotations
│   └── images
├── mixed_grounding
│   ├── annotations
│   ├── images
├── mixed_grounding
│   ├── annotations
│   ├── images
├── objects365v1
│   ├── annotations
│   ├── train
│   ├── val
```

## 2.3 数据集下载介绍与下载

| 数据集名称   | 领域      |    类别数    | 图片数量 | 说明                                                    |
| :----------- | :-------- | :----------: | :------: | :------------------------------------------------------ |
| Objects365v1 | detection |     365      |    2M    | 和传统的目标检测数据集一样                              |
| GQA          | grounding | 无明确的类别 |   148k   | 是一个用于问答的数据集                                  |
| Flickr30k    | grounding | 无明确的类别 |   31k    | 每张图片都有5个captions和一系列的bbox（实体版才有bbox） |
| LVIS         | grounding |     1203     |   160k   | 每个类别都会有一个描述语句                              |

### 2.3.1 Objects365v1

- 论文链接：[Objects365: A Large-scale, High-quality Dataset for Object Detection](http://openaccess.thecvf.com/content_ICCV_2019/papers/Shao_Objects365_A_Large-Scale_High-Quality_Dataset_for_Object_Detection_ICCV_2019_paper.pdf)

- 说明：Objects365就是一个传统的目标检测数据集，图片/类别没有对应的描述性文本，但单独的类别可以有近义词。

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-06-14-16-49-56.png
    width=75%></br><center>Objects365v1数据集示例</center>
</div></br>

### 2.3.2 GQA

- 论文链接：[GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering](https://arxiv.org/abs/1902.09506v3)
- 说明：GQA数据集其实不是用于目标检测的，它本质上是一个问答数据集，就是说它的每一张图片都有一些列问题以及对应的答案。

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-06-14-16-56-16.png
    width=60%></br><center>GQA数据集示例</center>
</div></br>

### 2.3.3 Flickr30k

- 论文链接：[Flickr30k Entities: Collecting Region-to-Phrase Correspondences for Richer Image-to-Sentence Models](https://arxiv.org/abs/1505.04870)

- 说明：原始的Flickr30k数据集包含了31,783张图片，每张图片配有5个由人类标注者提供的参考句子（captions），共计158,915个句子。随后，有研究者为了进一步提升数据集的多模态研究价值，在Flickr30k数据集的基础上进行了扩展，创建了Flickr30k Entities数据集。Flickr30k Entities数据集在原有的图片和句子的基础上增加了244,000个共指链（coreference chains）和276,000个手动标注的边界框（bounding boxes）。这些边界框与图片中提及的实体相对应，极大地丰富了数据集的语义信息，为图像描述、视觉问答等任务提供了更丰富的标注资源。

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-06-14-16-32-48.png
    width=80%></br><center>Flickr30k entities数据集示例</center>
</div></br>

### 2.3.4 LVIS

- 论文链接：[LVIS: A Dataset for Large Vocabulary Instance Segmentation](https://arxiv.org/abs/1908.03195)
- 说明：LVIS和传统的目标检测数据集不同的是：
  - LVIS每个object的坐标是多边形
  - LVIS为每个类别都定义了一段文字用来描述该类别
  - 单独的类别可以有近义词

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-06-14-17-05-06.png
    width=40%></br><center></center>
</div></br>

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-06-14-17-02-30.png
    width=70%></br><center></center>
</div></br>

### 2.3.5 数据集下载地址

| 数据集名称 | 图片下载地址 | 标签下载地址 |
| :--- | :------| :-------------- |
| Objects365v1 | [Objects365 train](https://opendatalab.com/OpenDataLab/Objects365_v1) | [objects365_train.json](https://opendatalab.com/OpenDataLab/Objects365_v1) |
| MixedGrounding | [GQA](https://nlp.stanford.edu/data/gqa/images.zip) | [final_mixed_train_no_coco.json](https://huggingface.co/GLIPModel/GLIP/blob/main/mdetr_annotations/final_mixed_train_no_coco.json) |
| Flickr30k | [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) |[final_flickr_separateGT_train.json](https://huggingface.co/GLIPModel/GLIP/blob/main/mdetr_annotations/final_flickr_separateGT_train.json) |
| LVIS-minival | [COCO val2017](https://cocodataset.org/) | [lvis_v1_minival_inserted_image_name.json](https://huggingface.co/GLIPModel/GLIP/blob/main/lvis_v1_minival_inserted_image_name.json) |


## 2.4 数据集类别

> 对于在Close-set（传统的目标检测数据集）目标检测上微调YOLO-World，建议使用多模态数据集。

### 2.4.1 设置类别

如果我们使用 `COCO-format` 自定义数据集，则“不需要”为自定义词汇表/类别定义数据集类。通过 `metainfo=dict(classes=your_classes)`, 在配置文件中显式设置 `CLASSES` 很简单：

```python
coco_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=your_classes),  # 这里需要填写具体的类别数
        data_root='data/your_data',  # 数据集ROOT
        ann_file='annotations/your_annotation.json',  # 标签文件（相对ROOT的路径）
        data_prefix=dict(img='images/'),  # 图片所在文件名称（相对ROOT的路径）
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/your_class_texts.json',  # 类别对应的文本文件路径（绝对路径）
    pipeline=train_pipeline)
```

为了训练YOLO-World，我们主要采用两种数据集类：

1. `MultiModalDataset`：数据集还是传统的目标检测数据集（转换为COCO格式即可），只不过需要为每个类别添加一个描述，即多了一个`class_text`文件
2. `YOLOv5MixedGroundingDataset`：数据集格式还是COCO格式，但<font color='red'><b></b></font>每个图片都会有一个文字描述（caption）。

下面我们详细说明一下这二者。

### 2.4.2 多模态数据集（MultiModalDataset）

`MultiModalDataset` 是预定义数据集类的简单包装器，例如 Objects365或COCO ，它将文本（类别文本）添加到数据集实例中以进行格式化输入文本。

`.json` 文件格式如下：

```json
[
    ["A_1","A_2"],
    ["B"],
    ["C_1", "C_2", "C_3"],
    ["..."]
]
```

其中：
- `"A_1"`和`"A_2"`是一个类别，二者为近义词，可以表示这一个类别。
- `"B"`：是一个类别，它没有近义词。
- `"C_1", "C_2", "C_3"`是一个类别，三者为近义词，均表示这个类别。

#### 1. LVIS的json文件格式示意（有近义词）：

```json
[
    [
        "aerosol can",
        "spray can"
    ],
    [
        "air conditioner"
    ],
    [
        "airplane",
        "aeroplane"
    ],
    [
        "alarm clock"
    ],
    [
        "alcohol",
        "alcoholic beverage"
    ],
    [
        "...",
        "有近义词！！！！！！"
    ],
    [
        "yogurt",
        "yoghurt",
        "yoghourt"
    ],
    [
        "yoke",
        "yoke animal equipment"
    ],
    [
        "zebra"
    ],
    [
        "zucchini",
        "courgette"
    ]
]
```

#### 2. COCO的json文件格式示意（没有近义词）：

```json
[
    [
        "person"
    ],
    [
        "bicycle"
    ],
    [
        "car"
    ],
    [
        "motorcycle"
    ],
    [
        "airplane"
    ],
    [
        "没有近义词！！！！！"
    ],
    [
        "teddy bear"
    ],
    [
        "hair drier"
    ],
    [
        "toothbrush"
    ]
]
```

#### 3. Object365V1的json文件格式示意（有近义词）：

```json
[
    [
        "person"
    ],
    [
        "sneakers"
    ],
    [
        "chair"
    ],
    [
        "hat"
    ],
    [
        "lamp"
    ],
    [
        "bottle"
    ],
    [
        "cabinet",
        "shelf"
    ],
    [
        "...",
        "有近义词！！！！！"
    ],
    [
        "iron"
    ],
    [
        "flashlight"
    ]
]
```

### 2.4.3 混合文本对齐数据集（YOLOv5MixedGroundingDataset）

`YOLOv5MixedGroundingDataset` 通过支持从 `json` 文件加载文本/标题来扩展 `COCO` 数据集。它是为 `MixedGrounding`或`Flickr30K` 设计的，<font color='red'><b>每个对象都有文本标记</b></font>。

### 2.4.4 自定义数据集

对于自定义数据集，建议根据用途转换注释文件。

> 💡 <font color='red'><b>请注意，基本上都需要将标签文件转换为标准 COCO 格式</b></font>。

1. 自定义词汇表（已修复）：可以采用 `MultiModalDataset` 包装器（wrapper）作为 `Objects365` 并为自定义类别创建`文本json`。
2. 大词汇量、基础、参考：请遵循 `MixedGrounding` 数据集的注释格式，其中添加 `caption`和`tokens_positive` 来<font color='red'><b>为每个对象分配文本</b></font>。<font color='blue'><b>文本可以是类别或名词短语</b></font>。-

### 2.4.5 CC3M 伪注释（Pseudo Annotations）

以下注释是根据论文中的自动标记过程生成的，然后根据这些注释报告结果。要使用CC3M注释，需要先准备 CC3M 图像。

| Data | Images | Boxes | File |
| :--: | :----: | :---: | :---: |
| CC3M-246K | 246,363 | 820,629 | [Download 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/cc3m_pseudo_annotations.json) |
| CC3M-500K | 536,405 | 1,784,405| [Download 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/cc3m_pseudo_500k_annotations.json) |
| CC3M-750K | 750,000 | 4,504,805 | [Download 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/cc3m_pseudo_750k_annotations.json) |

CC3M的标注示例如下：

<details><summary>🪐 展开查看CC3M的标签内容</summary>

```json
{
    "categories": {
        {"name": "appropriate attire", "id": 1},
        {"name": "golfers", "id": 2},
        {"name": "the exhibit", "id": 3},
        {"name": "art museum", "id": 4},
        {"name": "homes", "id": 5},
        {"name": "the finish line", "id": 6},
        {"name": "a runner", "id": 7},
        {"name": "actor", "id": 8},
        {"name": "psych folk artist performs", "id": 9},
        {"name": "the opening night", "id": 10},
    },
    "images": {
        {"id": 0, "file_name": "2916026_474308128", "height": 811, "width": 1024, "caption": "actor , who recently played his first round of golf , feels rules on appropriate attire for golfers are a good idea ."},
        {"id": 1, "file_name": "426852_4023546009", "height": 400, "width": 600, "caption": "the exhibit displays homes commissioned by art museum ."},
        {"id": 2, "file_name": "429649_2882962956", "height": 706, "width": 1024, "caption": "a runner crosses the finish line during recurring competition ."},
        {"id": 3, "file_name": "2906408_1262177123", "height": 447, "width": 640, "caption": "pop artist , film director and actor ."},
        {"id": 4, "file_name": "2115477_1264179008", "height": 612, "width": 408, "caption": "psych folk artist performs at festival"},
        {"id": 5, "file_name": "1665090_1070062237", "height": 681, "width": 1024, "caption": "musical artist performs on stage supporting artist on the opening night of their tour ."},
        {"id": 6, "file_name": "440715_513854996", "height": 900, "width": 600, "caption": "this dress just reminds me so much of one of my bridesmaids !"},
        {"id": 7, "file_name": "443433_3084953150", "height": 438, "width": 640, "caption": "rhythm and blues artist performs as part of the event"},
        {"id": 8, "file_name": "2078134_958326625", "height": 447, "width": 640, "caption": "the village of person in the island"},
        {"id": 9, "file_name": "1263157_4047346620", "height": 408, "width": 612, "caption": "actor arrives at the premiere"},
        {"id": 10, "file_name": "2929378_677160606", "height": 488, "width": 640, "caption": "politician on a state visit visited country and met with religious leader"},
    }, 
    "annotations": {
        {"image_id": 0, "id": 0, "category_id": 1, "area": 422653.1869565472, "bbox": [0.0, 379.05767822265625, 976.2655029296875, 432.92852783203125], "iscrowd": 0, "score": 0.3347477614879608, "tokens": "appropriate attire"},
        {"image_id": 0, "id": 1, "category_id": 2, "area": 790882.0195608903, "bbox": [0.0, 11.648781776428223, 989.0121459960938, 799.668662071228], "iscrowd": 0, "score": 0.3464314341545105, "tokens": "golfers"},
        {"image_id": 1, "id": 2, "category_id": 3, "area": 130493.92645893525, "bbox": [129.6766357421875, 4.1426849365234375, 332.00677490234375, 393.0459747314453], "iscrowd": 0, "score": 0.3408966362476349, "tokens": "the exhibit"},
        {"image_id": 1, "id": 3, "category_id": 3, "area": 18324.43645722326, "bbox": [0.7921218872070312, 125.02227783203125, 109.76447296142578, 166.9432373046875], "iscrowd": 0, "score": 0.3054649829864502, "tokens": "the exhibit"},
        {"image_id": 1, "id": 4, "category_id": 4, "area": 238566.77633925015, "bbox": [0.240753173828125, 0.5902252197265625, 599.0253601074219, 398.2582244873047], "iscrowd": 0, "score": 0.3036207854747772, "tokens": "art museum"},
        {"image_id": 1, "id": 5, "category_id": 5, "area": 238566.77633925015, "bbox": [0.240753173828125, 0.5902252197265625, 599.0253601074219, 398.2582244873047], "iscrowd": 0, "score": 0.3074909448623657, "tokens": "homes"},
        {"image_id": 2, "id": 6, "category_id": 6, "area": 328616.3230983538, "bbox": [255.7866668701172, 18.506240844726562, 502.07691955566406, 654.5139007568359], "iscrowd": 0, "score": 0.3392699360847473, "tokens": "the finish line"},
        {"image_id": 2, "id": 7, "category_id": 7, "area": 65720.81522059813, "bbox": [247.374755859375, 270.8210754394531, 162.9920654296875, 403.2148132324219], "iscrowd": 0, "score": 0.34287354350090027, "tokens": "a runner"},
        {"image_id": 2, "id": 8, "category_id": 7, "area": 313720.4859452478, "bbox": [281.114990234375, 16.223989486694336, 478.8563232421875, 655.1453342437744], "iscrowd": 0, "score": 0.30990859866142273, "tokens": "a runner"},
        {"image_id": 2, "id": 9, "category_id": 7, "area": 96920.24000985548, "bbox": [429.16864013671875, 87.87799072265625, 176.37725830078125, 549.5053100585938], "iscrowd": 0, "score": 0.3093494474887848, "tokens": "a runner"},
        {"image_id": 3, "id": 10, "category_id": 8, "area": 47870.292117924895, "bbox": [226.57835388183594, 104.74246215820312, 147.6943817138672, 324.1172180175781], "iscrowd": 0, "score": 0.3774438202381134, "tokens": "actor"},
        {"image_id": 3, "id": 11, "category_id": 8, "area": 49782.2714155633, "bbox": [342.3399353027344, 93.310546875, 148.91860961914062, 334.29180908203125], "iscrowd": 0, "score": 0.3582227826118469, "tokens": "actor"},
        {"image_id": 3, "id": 12, "category_id": 8, "area": 42798.63825566019, "bbox": [113.33319091796875, 110.71199035644531, 134.9569549560547, 317.12806701660156], "iscrowd": 0, "score": 0.34423449635505676, "tokens": "actor"},
        {"image_id": 4, "id": 13, "category_id": 9, "area": 158787.37313683218, "bbox": [28.66830825805664, 80.38202667236328, 298.2539939880371, 532.389762878418], "iscrowd": 0, "score": 0.370086133480072, "tokens": "psych folk artist performs"},
        {"image_id": 5, "id": 14, "category_id": 10, "area": 690727.2885092197, "bbox": [3.4731175899505615, 0.566997766494751, 1020.6946680545807, 676.722736120224], "iscrowd": 0, "score": 0.30705296993255615, "tokens": "the opening night"},
        {"image_id": 5, "id": 15, "category_id": 11, "area": 377094.0525847074, "bbox": [287.8883972167969, 16.25448989868164, 569.5247497558594, 662.1205711364746], "iscrowd": 0, "score": 0.33824190497398376, "tokens": "musical artist performs"},
        {"image_id": 5, "id": 16, "category_id": 12, "area": 377094.0525847074, "bbox": [287.8883972167969, 16.25448989868164, 569.5247497558594, 662.1205711364746], "iscrowd": 0, "score": 0.31286439299583435, "tokens": "artist"},
        {"image_id": 6, "id": 17, "category_id": 13, "area": 361252.4822749605, "bbox": [56.020660400390625, 67.7717514038086, 473.9666442871094, 762.1896743774414], "iscrowd": 0, "score": 0.39791786670684814, "tokens": "bridesmaids"},
        {"image_id": 6, "id": 18, "category_id": 14, "area": 111691.20079200249, "bbox": [190.32395935058594, 241.3621826171875, 186.20994567871094, 599.8132934570312], "iscrowd": 0, "score": 0.3924933671951294, "tokens": "this dress"},
        {"image_id": 6, "id": 19, "category_id": 14, "area": 262045.76735822484, "bbox": [87.693359375, 79.24378967285156, 344.794189453125, 760.0063323974609], "iscrowd": 0, "score": 0.31617414951324463, "tokens": "this dress"},
        {"image_id": 7, "id": 20, "category_id": 15, "area": 176702.29679879732, "bbox": [159.9798583984375, 53.910037994384766, 481.47216796875, 367.00417709350586], "iscrowd": 0, "score": 0.34347665309906006, "tokens": "performs"},
        {"image_id": 7, "id": 21, "category_id": 16, "area": 26940.760472631548, "bbox": [90.44544982910156, 253.2623291015625, 231.88111877441406, 116.18350219726562], "iscrowd": 0, "score": 0.30747899413108826, "tokens": "part"},
        {"image_id": 7, "id": 22, "category_id": 16, "area": 4047.9430108852684, "bbox": [572.8274536132812, 314.28662109375, 68.55877685546875, 59.04339599609375], "iscrowd": 0, "score": 0.3061348795890808, "tokens": "part"},
        {"image_id": 8, "id": 23, "category_id": 17, "area": 498.5000101849437, "bbox": [618.4949340820312, 280.6375732421875, 19.456787109375, 25.620880126953125], "iscrowd": 0, "score": 0.31343746185302734, "tokens": "person"},
        {"image_id": 8, "id": 24, "category_id": 18, "area": 84702.77219054895, "bbox": [88.44400024414062, 160.23072814941406, 549.9750061035156, 154.0120391845703], "iscrowd": 0, "score": 0.3755825161933899, "tokens": "the village"},
        {"image_id": 8, "id": 25, "category_id": 19, "area": 136346.31888543777, "bbox": [159.03176879882812, 1.7530081272125244, 477.5589904785156, 285.5067574977875], "iscrowd": 0, "score": 0.31633585691452026, "tokens": "the island"},
        {"image_id": 9, "id": 26, "category_id": 8, "area": 190156.06273768027, "bbox": [96.15538787841797, 6.901155471801758, 474.92542266845703, 400.3914165496826], "iscrowd": 0, "score": 0.34347039461135864, "tokens": "actor"},
        {"image_id": 9, "id": 27, "category_id": 20, "area": 231256.54143596182, "bbox": [20.39750862121582, 6.0372772216796875, 577.8034801483154, 400.23390197753906], "iscrowd": 0, "score": 0.35184529423713684, "tokens": "the premiere"},
        {"image_id": 10, "id": 28, "category_id": 21, "area": 275776.2488502312, "bbox": [21.93243980407715, 12.839357376098633, 610.1771793365479, 451.96093559265137], "iscrowd": 0, "score": 0.3917006850242615, "tokens": "a state visit"},
        {"image_id": 10, "id": 29, "category_id": 22, "area": 95793.32920437756, "bbox": [85.48125457763672, 24.463964462280273, 218.02680206298828, 439.36492347717285], "iscrowd": 0, "score": 0.3641899526119232, "tokens": "politician"},
        {"image_id": 10, "id": 30, "category_id": 23, "area": 103885.73604834673, "bbox": [367.5455017089844, 50.02199172973633, 258.4236145019531, 401.9978446960449], "iscrowd": 0, "score": 0.4383631646633148, "tokens": "religious leader"},
    }
}
```

</details>

首先我们可以看到，CC3M的标签格式和COCO是一样的，但有一点细微的差别：
- 对于`"categories"`，相比原本的COCO少了一个`"supercategory"`键值对，这个影响不大。
- 对于`"images"`，相比原本的COCO多了`"caption"`，这个是整张图片的标注。
- 对于`"annotations"`，相比原版的COCO多了`"score"`和`"tokens"`
  - `"score"`：CLIP模型给对象-tokens的匹配度得分
  - `"tokens"`：当前对象的最简单描述（名词）

# 3. 模型训练与评估（Training & Evaluation）

YOLO-World采用MMYOLO默认的训练或评估脚本。在 `configs/pretrain`和`configs/finetune_coco` 中提供了用于预训练和微调的配置。

| 配置文件名称                                                               | 版本  | 模型规格 |      用途      | 文本-图像连接方式 | 分割标签细化 | 优化器 | Epochs |  LR   |
| :------------------------------------------------------------------------- | :---: | :------: | :------------: | :---------------: | :----------: | :----: | :----: | :---: |
| yolo_world_l_dual_vlpan_2e-4_80e_8gpus_finetune_coco.py                    |  v1   |    L     |    微调COCO    |    Dual VL-PAN    |      -       | AdamW  |   80   | 2e-4  |
| yolo_world_l_dual_vlpan_2e-4_80e_8gpus_mask-refine_finetune_coco.py        |  v1   |    L     |    微调COCO    |    Dual VL-PAN    |      √       | AdamW  |   80   | 2e-4  |
| yolo_world_l_efficient_neck_2e-4_80e_8gpus_mask-refine_finetune_coco.py    |  v1   |    L     |    微调COCO    |  Efficient Neck   |      √       | AdamW  |   80   | 2e-4  |
| yolo_world_v2_l_efficient_neck_2e-4_80e_8gpus_mask-refine_finetune_coco.py |  v2   |    L     |    微调COCO    |  Efficient Neck   |      √       | AdamW  |   80   | 2e-4  |
| yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py       |  v2   |    L     |    微调COCO    |      VL-PAN       |      √       | AdamW  |   80   | 2e-4  |
| yolo_world_v2_l_vlpan_bn_sgd_1e-3_40e_8gpus_finetune_coco.py               |  v2   |    L     |    微调COCO    |      VL-PAN       |      -       |  SGD   |   40   | 1e-3  |
| yolo_world_v2_l_vlpan_bn_sgd_1e-3_80e_8gpus_mask-refine_finetune_coco.py   |  v2   |    L     |    微调COCO    |      VL-PAN       |      √       |  SGD   |   80   | 1e-3  |
| yolo_world_v2_m_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py       |  v2   |    M     |    微调COCO    |      VL-PAN       |      √       | AdamW  |   80   | 2e-4  |
| yolo_world_v2_s_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py             |  v2   |    S     |    微调COCO    |         -         |      √       | AdamW  |   80   | 2e-4  |
| yolo_world_v2_s_rep_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py   |  v2   |    S     |    微调COCO    |    Rep-VL-PAN     |      √       | AdamW  |   80   | 2e-4  |
| yolo_world_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py       |  v2   |    S     |    微调COCO    |      VL-PAN       |      √       | AdamW  |   80   | 2e-4  |
| yolo_world_v2_xl_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py      |  v2   |    XL    |    微调COCO    |      VL-PAN       |      √       | AdamW  |   80   | 2e-4  |
| yolo_world_v2_x_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py       |  v2   |    X     |    微调COCO    |      VL-PAN       |      √       | AdamW  |   80   | 2e-4  |
| yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_prompt_tuning_coco.py  |  v2   |    L     | 提示词微调COCO |      VL-PAN       |      √       | AdamW  |   80   | 2e-4  |
| yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_prompt_tuning_coco.py              |  v2   |    L     | 提示词微调COCO |      VL-PAN       |      -       | AdamW  |   80   | 2e-4  |
| yolo_world_v2_l_vlpan_bn_sgd_1e-3_80e_8gpus_all_finetuning_coco.py         |  v2   |    L     | 提示词微调COCO |      VL-PAN       |      -       |  SGD   |   80   | 1e-3  |

## 3.1 模型训练

训练 YOLO-World 很简单：

```bash
chmod +x tools/dist_train.sh

# sample command for pre-training, use AMP for mixed-precision training
./tools/dist_train.sh configs/pretrain/yolo_world_l_t2i_bn_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py 8 --amp
```

> 注意：YOLO-World在4 个节点上进行预训练，每个节点有 8个GPU（总共 32个GPU）。对于预训练，应指定多节点训练的 node_rank和nnodes 。

脚本内容如下：

```bash
#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${MASTER_PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
```

其中：
- `CONFIG=$1`和`GPUS=$2`表示读取脚本的第一个和第二参数分别给两个变量
- `NNODES=${NNODES:-1}`：设置 `NNODES` 变量，默认值为 1，表示单个节点
- `NODE_RANK=${NODE_RANK:-0}`：设置 `NODE_RANK` 变量，默认值为 0，表示节点的序号
- `PORT=${MASTER_PORT:-29500}`：设置 `PORT` 变量，默认值为 29500，表示使用的端口号
- `MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}`：设置 `MASTER_ADDR` 变量，默认值为 `"127.0.0.1"`，表示主节点的地址
- `PYTHONPATH="$(dirname $0)/..":$PYTHONPATH`：设置 `PYTHONPATH` 环境变量，添加当前脚本所在目录的上一级目录到 `PYTHONPATH`
- `python -m torch.distributed.launch`：调用 python 命令，使用 `torch.distributed.launch` 模块启动分布式训练
  - `--nnodes=$NNODES`：指定节点数量
  - `--node_rank=$NODE_RANK`：指定当前节点的序号
  - `--master_addr=$MASTER_ADDR`：指定主节点的地址
  - `--nproc_per_node=$GPUS`：指定每个节点上使用的 GPU 数量
  - `--master_port=$PORT`：指定通信端口
  - `$(dirname "$0")/train.py`：指定训练脚本的路径
  - `$CONFIG`：传递配置文件路径作为参数
  - `--launcher pytorch ${@:3}`  # 使用 pytorch 作为启动器，并传递脚本的其余参数

> `${@:3}` 表示从脚本的第四个参数开始，传递所有剩余的参数给 `train.py` 脚本

## 3.2 模型评估

评估 YOLO-World 也很简单：

```bash
chmod +x tools/dist_test.sh

./tools/dist_test.sh path/to/config path/to/weights 8
```

> 注：这里主要评估 LVIS-minival 预训练的性能。

## 3.3 找到best.pth

YOLO-World默认不会自动保存最佳权重，我们可以有两种方法来找到`best.pth`：

1. 在配置文件中添加自动保存最佳模型
2. 使用脚本查询最佳模型的epoch

### 3.3.1 在配置文件中添加自动保存最佳模型

将`save_best=None`修改为`save_best='auto'`，如下所示：

```python
default_hooks = dict(param_scheduler=dict(scheduler_type='linear',
                                          lr_factor=0.01,
                                          max_epochs=max_epochs),
                     checkpoint=dict(max_keep_ckpts=-1,
                                     save_best=None,  # 💡 将save_best=None修改为save_best='auto'
                                     interval=save_epoch_intervals))
```

### 3.3.2 使用脚本查询最佳模型的epoch

创建一个名为`find_best_epoch.py`的文件，将下面内容粘贴进去：

```python
import json
from pathlib import Path


# ============================== 参数 ==============================
src_dir = 'work_dirs/yolo_world_v2_s_vlpan_bn_2e-4_80e_1gpus-refine_finetune'  # 💡 模型所在文件夹路径
dst_file_name = 'scalars.json'  # 要使用的json文件名（建议不改）
# ==================================================================

src_dir = Path(src_dir)
if src_dir.is_file():
    src_dir = src_dir.parent

json_paths = [json_path for json_path in src_dir.rglob(dst_file_name)]

assert len(json_paths) == 1, f"❌ There are either 0 or more than one {dst_file_name} files present. \
    Please specify a more detailed directory path to avoid conflicts."

json_path = json_paths[0]
dst_path = json_path.parent.joinpath('best_epoch.txt')

with json_path.open('r') as f:
    lines = f.readlines()
    
lines = [line.strip() for line in lines]
lines = [json.loads(line) for line in lines]

# 找到最大的epoch
best_mAP50 = 0
best_dict = {}
for line in lines:
    if line.get('coco/bbox_mAP_50', -1) > best_mAP50:
        best_mAP50 = line['coco/bbox_mAP_50']
        best_dict = line

if best_dict:
    with dst_path.open('w') as f:
        for k, v in best_dict.items():
            f.write(f"{k}: {v}\n")
    print(f"✅ The 'best_epoch.txt' saves in {str(dst_path)}")
else:
    print(f"❌ Best epoch not found!")
```

运行该脚本后，结果如下：

```
(yolo-world) root@xxxxx:/home/Le0v1n/code/YOLO-World# python find_best_epoch.py
✅ The 'best_epoch.txt' saves in work_dirs/yolo_world_v2_s_vlpan_bn_2e-4_80e_1gpus-refine_finetune/20240618_xxxxxx/vis_data/best_epoch.txt
```

所有的结果都保存在了`scalars.json`所在文件夹的`best_epoch.txt`文件中，内容如下：

```
coco/bbox_mAP: 0.648
coco/bbox_mAP_50: 0.884
coco/bbox_mAP_75: 0.691
coco/bbox_mAP_s: 0.194
coco/bbox_mAP_m: 0.5
coco/bbox_mAP_l: 0.797
data_time: 0.000408179329528557
time: 0.023147689028407362
step: 80  # 💡 这个就是最佳的epoch数
```

### 3.3.3 FAQ

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：使用第二种方法找到的epoch没有被保存怎么办？
🥳 𝑨𝒏𝒔𝒘𝒆𝒓：YOLO-World默认是每5个epoch保存一个权重，所以可能会出现这样的问题，那么我们只能按照最佳epoch找到最近的epoch权重文件作为`best.pth`。

# 4. 微调（Fine-tuning）

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-06-17-10-39-25.png
    width=60%></br><center>选择作者预先训练好的YOLO-World权重（ckpt）并对其进行微调！</center>
</div></br>

YOLO-World 支持零样本推理和三种类型的微调方法：(1) 普通微调，(2) 提示微调，(3) 重参数化微调。

## 4.1 普通微调（normal fine-tuning）

### 4.1.1 微调要求

微调 YOLO-World 很便宜：
- 它不需要32个GPU来进行多节点分布式训练。 8个甚至1个GPU就足够了。
- 它不需要很长的时间安排，例如训练YOLOv5或YOLOv8需要300个epoch或500个epoch。考虑到原作者提供了良好的预训练权重，<font color='red'><b>80个或更少的epoch就足够了</b></font>。

### 4.1.2 数据准备

微调数据集应具有与预训练数据集类似的格式（即COCO的格式）。可以参考 [第二章的内容](#Title_2) 了解有关如何构建数据集的更多详细信息：

- 如果我们对 YOLO-World 进行微调以进行闭集/自定义词汇对象检测（close-set / custom vocabulary object detection），则首选使用 `MultiModalDataset`和`text json`。
- 如果我们对 YOLO-World 进行微调以进行富文本（rich texts）或grounding任务（grounding tasks）的开放词汇检测，则首选使用 `MixedGroundingDataset`。

### 4.1.3 超参数和配置文件（Hyper-parameters and Config）

#### 1. 基本配置文件（Basic config file）

如果微调数据集包含mask注释：

```python
_base_ = ('../../third_party/mmyolo/configs/yolov8/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py')
```

如果微调数据集不包含mask注释：

```python
_base_ = ('../../third_party/mmyolo/configs/yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco.py')
```

> 💡 这里的mask注释的意思是，数据集中是否有分割标注坐标，如果有则包含mask注释，否则不包含mask注释。

#### 2. 训练策略（Training Schemes）

减少 epoch 并调整学习率：

```python
max_epochs = 80
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 16
close_mosaic_epochs=10

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        _base_.val_interval_stage2)])
```

#### 3. 数据集

```python
coco_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='data/coco',
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/coco_class_texts.json',
    pipeline=train_pipeline)
```

### 4.1.4 🚀 无需 RepVL-PAN 或文本编码器进行微调

为了进一步提高效率和简单性，我们可以在没有 `RepVL-PAN` 和文本编码器的情况下微调 YOLO-World 的高效版本。 YOLO-World 的高效版本与原始 YOLOv8 具有相似的架构或层，但作者提供了大规模数据集上的预训练权重。预训练的YOLO-World具有很强的泛化能力，与在COCO数据集上训练的YOLOv8相比更加鲁棒。更多详细信息可以参考`configs/finetune_coco/yolo_world_v2_l_efficient_neck_2e-4_80e_8gpus_mask-refine_finetune_coco.py`。

> 高效的YOLO-World采用 `EfficientCSPLayerWithTwoConv`，并且可以<font color='blue'><b>在推理或导出模型时删除文本编码器</b></font>。

> 采用这种方式就相当于不再使用Text Embedding这些东西，而是直接把YOLO-World当做YOLOv8那样使用。

```python
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='EfficientCSPLayerWithTwoConv')))
```

#### 6. 完整的配置文件

<details><summary>🪐 点击展开查看完整的配置文件</summary>

```python
_base_ = (
    '../../third_party/mmyolo/configs/yolov8/'
    'yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(
    imports=['yolo_world'],
    allow_failed_imports=False)

# hyper-parameters
num_classes = 80
num_training_classes = 80
max_epochs = 80  # Maximum training epochs
close_mosaic_epochs = 10
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 16
# 权重下载链接：https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_obj365v1_goldg_cc3mlite_pretrain-ca93cd1f.pth
load_from = 'pretrained_models/yolo_world_v2_l_obj365v1_goldg_cc3mlite_pretrain-ca93cd1f.pth'
text_model_name = '../pretrained_models/clip-vit-base-patch32-projection'
text_model_name = 'openai/clip-vit-base-patch32'
persistent_workers = False

# model settings
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model_name,
            frozen_modules=['all'])),
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='EfficientCSPLayerWithTwoConv')),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    use_bn_head=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# dataset settings
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]
mosaic_affine_transform = [
    dict(
        type='MultiModalMosaic',
        img_scale=_base_.img_scale,
        pad_val=114.0,
        pre_transform=_base_.pre_transform),
    dict(type='YOLOv5CopyPaste', prob=_base_.copypaste_prob),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(1 - _base_.affine_scale,
                             1 + _base_.affine_scale),
        # img_scale is (width, height)
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114),
        min_area_ratio=_base_.min_area_ratio,
        use_mask_refine=_base_.use_mask2refine)
]
train_pipeline = [
    *_base_.pre_transform,
    *mosaic_affine_transform,
    dict(
        type='YOLOv5MultiModalMixUp',
        prob=_base_.mixup_prob,
        pre_transform=[*_base_.pre_transform,
                       *mosaic_affine_transform]),
    *_base_.last_transform[:-1],
    *text_transform
]
train_pipeline_stage2 = [
    *_base_.train_pipeline_stage2[:-1],
    *text_transform
]
coco_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='data/coco',
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/coco_class_texts.json',
    pipeline=train_pipeline)

train_dataloader = dict(
    persistent_workers=persistent_workers,
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    dataset=coco_train_dataset)
test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param', 'texts'))
]
coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='data/coco',
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/coco_class_texts.json',
    pipeline=test_pipeline)
val_dataloader = dict(dataset=coco_val_dataset)
test_dataloader = val_dataloader
# training settings
default_hooks = dict(
    param_scheduler=dict(
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        max_keep_ckpts=-1,
        save_best=None,
        interval=save_epoch_intervals))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        _base_.val_interval_stage2)])
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(
        custom_keys={'backbone.text_model': dict(lr_mult=0.01),
                     'logit_scale': dict(weight_decay=0.0)}),
    constructor='YOLOWv5OptimizerConstructor')

# evaluation settings
val_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='data/coco/annotations/instances_val2017.json',
    metric='bbox')
```
</details>

### 4.1.5 启动微调

```bash
./dist_train.sh <配置文件路径> <NUM_GPUS> --amp
```

例子：

```bash
CUDA_VISIBLE_DEVICES=1,2 ./dist_train.sh configs/finetune_coco/yolo_world_v2_s_vlpan_bn_2e-4_80e_1gpus-refine_finetune.py 2 --amp
```

### 4.1.6 🔥 普通微调示例

#### 1. 数据集转换 <a id=4.1.6.1></a>

我们有一个类似于coco128的数据集，它的目录结构如下：

```
data/coco128
├── test
│   ├── images
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   └── labels
│   │   ├── 000001.txt
│   │   ├── 000002.txt
│   │   └── ...
├── train
│   ├── images
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   └── labels
│   │   ├── 000001.txt
│   │   ├── 000002.txt
│   │   └── ...
└── val
    ├── images
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ...
    └── labels
        ├── 000001.txt
        ├── 000002.txt
        └── ...
```

我们需要在`train`、`val`、`test`文件夹中都添加一个类别文件：`classes.txt`，内容如下所示：

```
person, bicycle, car, motorcycle,...
```

之后将数据集转换为COCO2017的样式，执行如下脚本：

```bash
python third_party/mmyolo/tools/dataset_converters/yolo2coco.py data/coco128/train
python third_party/mmyolo/tools/dataset_converters/yolo2coco.py data/coco128/val
python third_party/mmyolo/tools/dataset_converters/yolo2coco.py data/coco128/test

mkdir data/coco128/annotations
cp data/coco128/train/annotations/result.json data/coco128/annotations/train.json
cp data/coco128/val/annotations/result.json data/coco128/annotations/val.json
cp data/coco128/test/annotations/result.json data/coco128/annotations/test.json
```

执行完毕后，目录结构如下所示：

```
data/coco128
├── annotations
│   ├── test.json   # 测试集标签文件
│   ├── train.json  # 训练集标签文件
│   └── val.json    # 验证集标签文件
├── test
│   ├── annotations
│   │   └── result.json
│   ├── classes.txt
│   ├── images
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   └── labels
│   │   ├── 000001.txt
│   │   ├── 000002.txt
│   │   └── ...
├── train
│   ├── annotations
│   │   └── result.json
│   ├── classes.txt
│   ├── images
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   └── labels
│   │   ├── 000001.txt
│   │   ├── 000002.txt
│   │   └── ...
└── val
    ├── annotations
    │   └── result.json
    ├── classes.txt
    ├── images
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ...
    └── labels
        ├── 000001.txt
        ├── 000002.txt
        └── ...
```

之后我们需要创建一个`text json`，位置为：`data/texts/coco128_class_texts.json`，内容如下：

```json
[
    [
        "person"
    ],
    [
        "bicycle"
    ],
    [
        "car"
    ],
    [
        "motorcycle"
    ],
    [
        "..."
    ]
]
```

依次描述数据集中每个类别。

#### 2. 创建配置文件

这里我们选用：`yolo_world_v2_s_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py`这个配置文件，因为我们没有提示词，所以不需要使用VL-PATH。

我们需要复制这个配置文件，然后重命名：

```bash
cp configs/finetune_coco/yolo_world_v2_s_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py configs/finetune_coco/yolo_world_v2_s_bn_2e-4_80e_1gpus_finetune_coco128.py
```

这里我们使用一个GPU，且标签中没有`mask`，所以我们在命名时把它们删掉了。但是我们发现了一个问题：这个配置文件对应的`.pth`权重文件作者并没有提供，所以要想使用这个配置文件，我们必须把预训练权重这行代码注释掉，这就相当于是从头开始训练了而非微调了。

为了解决这个问题，我们只能使用其他的配置文件。这里我选择了`yolo_world_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py`这个配置文件，这个配置文件和`yolo_world_v2_s_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py`的不同在于后者使用了VL-PAN和mask-refine。

- 因为使用了VL-PAN，所以我们需要一个文本文件，这个文件我们其实刚刚已经定义好了，即`data/texts/coco128_class_texts.json`。
- 由于它也使用了mask-refine，所以我们需要在配置文件中关闭它（这里其实不用关也可以😂）。

我们需要把配置文件复制一份，然后修改它：

```bash
cp configs/finetune_coco/yolo_world_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py configs/finetune_coco/yolo_world_v2_s_vlpan_bn_2e-4_80e_1gpus_finetune_coco128.py
```

该配置文件内容如下：

<details><summary>🪐 点击查看完整的配置文件</summary>

```python
_base_ = (
    '../../third_party/mmyolo/configs/yolov8/'
    'yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(
    imports=['yolo_world'],
    allow_failed_imports=False)

# hyper-parameters
num_classes = 80
num_training_classes = 80
max_epochs = 80  # Maximum training epochs
close_mosaic_epochs = 10
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 16
# 权重下载链接：https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth
load_from = 'pretrained_models/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth'
text_model_name = '../pretrained_models/clip-vit-base-patch32-projection'
text_model_name = 'openai/clip-vit-base-patch32'
persistent_workers = False
mixup_prob = 0.15
copypaste_prob = 0.3

# model settings
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model_name,
            frozen_modules=['all'])),
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    use_bn_head=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# dataset settings
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]
mosaic_affine_transform = [
    dict(
        type='MultiModalMosaic',
        img_scale=_base_.img_scale,
        pad_val=114.0,
        pre_transform=_base_.pre_transform),
    dict(type='YOLOv5CopyPaste', prob=copypaste_prob),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(1 - _base_.affine_scale,
                             1 + _base_.affine_scale),
        # img_scale is (width, height)
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114),
        min_area_ratio=_base_.min_area_ratio,
        use_mask_refine=False)  # 💡 这里我们不使用了mask-refine，直接设置为False
]
train_pipeline = [
    *_base_.pre_transform,
    *mosaic_affine_transform,
    dict(
        type='YOLOv5MultiModalMixUp',
        prob=mixup_prob,
        pre_transform=[*_base_.pre_transform,
                       *mosaic_affine_transform]),
    *_base_.last_transform[:-1],
    *text_transform
]
train_pipeline_stage2 = [
    *_base_.train_pipeline_stage2[:-1],
    *text_transform
]
coco_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='data/coco128',
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/coco128_class_texts.json',  # 💡 这里我们定义好类别的文本描述json文件
    pipeline=train_pipeline)

train_dataloader = dict(
    persistent_workers=persistent_workers,
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    dataset=coco_train_dataset)
test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param', 'texts'))
]
coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='data/coco128',
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/coco128_class_texts.json',
    pipeline=test_pipeline)
val_dataloader = dict(dataset=coco_val_dataset)
test_dataloader = val_dataloader
# training settings
default_hooks = dict(
    param_scheduler=dict(
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        max_keep_ckpts=-1,
        save_best=None,
        interval=save_epoch_intervals))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        _base_.val_interval_stage2)])
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(
        custom_keys={'backbone.text_model': dict(lr_mult=0.01),
                     'logit_scale': dict(weight_decay=0.0)}),
    constructor='YOLOWv5OptimizerConstructor')

# evaluation settings
val_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='data/coco128/annotations/val.json',
    metric='bbox')
```

</details>

#### 3. 开始微调

运行结果如下所示：

```
# 设置要使用的GPU索引
export CUDA_VISIBLE_DEVICES=1

# 调用脚本开始训练：
#     PARAM1: 具体的配置文件路径
#     PARAM2: 使用的GPU数量
#     PARAM3: 是否要开启AMP（AutoMixedPrecision，自动混合精度）
bash tools/dist_train.sh \
    configs/finetune_coco/yolo_world_v2_s_vlpan_bn_2e-4_80e_1gpus_finetune_coco128.py \
    1 \
    --amp
```

#### 4. 过程展示

```log
2024/06/19 01:21:38 - mmengine - INFO - Load checkpoint from pretrained_models/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth
2024/06/19 01:21:38 - mmengine - WARNING - "FileClient" will be deprecated in future. Please use io functions in https://mmengine.readthedocs.io/en/latest/api/fileio.html#file-io
2024/06/19 01:21:38 - mmengine - WARNING - "HardDiskBackend" is the alias of "LocalBackend" and the former will be deprecated in future.
2024/06/19 01:21:38 - mmengine - INFO - Checkpoints will be saved to /home/Le0v1n/code/YOLO-World/work_dirs/yolo_world_v2_s_vlpan_bn_2e-4_80e_1gpus_finetune_coco128.
2024/06/19 01:21:51 - mmengine - INFO - Epoch(train)  [1][  50/1509]  base_lr: 2.0000e-04 lr: 2.1648e-06  eta: 8:27:55  time: 0.2526  data_time: 0.0334  memory: 9182  grad_norm: nan  loss: 59.1782  loss_cls: 21.0711  loss_bbox: 18.8034  loss_dfl: 19.3038
2024/06/19 01:21:59 - mmengine - INFO - Epoch(train)  [1][ 100/1509]  base_lr: 2.0000e-04 lr: 4.3738e-06  eta: 7:02:30  time: 0.1678  data_time: 0.0038  memory: 4594  grad_norm: 439.5705  loss: 53.0932  loss_cls: 16.6841  loss_bbox: 17.5144  loss_dfl: 18.8946
```

#### 5. FAQ

如果在训练过程中出现报错，可以参考这位大佬写的博客😎：[本地及云服务器上部署yoloworld的过程中遇到一些问题整理记录](https://blog.csdn.net/ITdaka/article/details/138863017)

## 4.2 提示词微调（prompt tuning）

### 4.2.1 带embedding的简单 YOLO-World（Simple YOLO-World with Embeddings）

为了简化YOLO-World并摆脱语言模型，作者定义了一个新的基本检测器`SimpleYOLOWorldDetector`：

`SimpleYOLOWorldDetector` 支持提示词嵌入（prompt embeddings）作为输入，并且<font color='red'><b>不再包含语言模型！</b></font>现在，YOLO-World采用 embeddings 作为语言输入，嵌入支持几种：
- （1）来自语言模型的文本嵌入（text embeddings），例如CLIP<font color='red'><b>语言</b></font>编码器
- （2）来自视觉模型的图像嵌入（image embeddings），例如，CLIP<font color='red'><b>视觉</b></font>编码器
- （3）图像文本融合嵌入
- （4）随机嵌入

其中：
- (1)(2)(3)支持零样本（zero-shot）推理
- (1)(2)(3)(4)可以快速调整自定义数据。

> 🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：怎么理解“<u>为了简化YOLO-World并摆脱语言模型</u>”这句话？
> 🥳 𝑨𝒏𝒔𝒘𝒆𝒓：我们刚才在普通微调的时候，是不是需要指定一个json文件，里面是每个类别的captions，那么既然这个json文件是一个文本，那么YOLO-World就需要一个语言模型来处理这个文本。所以YOLO-World相当于是调用了另外一个模型来做这个事情。“<u>新的基本检测器SimpleYOLOWorldDetector</u>”意味着，YOLO-World在处理文本的时候不需要调用其他模型了，因为这个Detector要的不是一个文本，而是一个Text Embedding，即一个文本嵌入的向量（具体来说是一个`ndarray`对象）。有了`ndarray`对象，YOLO-World的VL-PAN模块就可以直接处理了（其实调用其他语言模型来处理文本也是想到获取一个`ndarray`对象😂）。

基本检测器`SimpleYOLOWorldDetector`定义如下：

```python
class SimpleYOLOWorldDetector(YOLODetector):
    """Implementation of YOLO World Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 **kwargs)
```

要以零样本（zero-shot）的方式使用它，我们需要预先计算文本嵌入text embeddings（图像嵌入image embeddings）并将其保存为具有 `NxD` 形状的 `numpy array (*.npy)` （其中，`N` 是提示词的数量，`D` 是嵌入的维度）。<font color='red'><b>目前，作者只支持一个类别拥有一个提示词</b></font>。我们可以对一类使用多个提示词，但需要在后处理步骤中合并结果。

### 4.2.2 提示词微调YOLO-World

作者对 YOLO-World 进行了即时调整，以保持零样本（zero-shot）能力，同时提高自定义数据集的性能。有关编写提示词调整配置的更多详细信息，我们可以参考 `configs/prompt_tuning_coco/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_prompt_tuning_coco.py`。

#### 1. 使用随机提示词

```python
dict(
    type='SimpleYOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    prompt_dim=text_channels,
    num_prompts=80,  # 一个类别有一个提示词
    ...
)
```

#### 2. 使用 CLIP 嵌入（文本、图像或文本图像嵌入）

> clip_vit_b32_coco_80_embeddings.npy 可以在 HuggingFace 下载，具体为：[clip_vit_b32_coco_80_embeddings.npy](https://huggingface.co/wondervictor/YOLO-World/blob/main/clip_vit_b32_coco_80_embeddings.npy)。

```python
dict(
    type='SimpleYOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    embedding_path='embeddings/clip_vit_b32_coco_80_embeddings.npy',
    prompt_dim=text_channels,
    num_prompts=80,
    ...
)
```

使用CLIP模型获取图像和文本嵌入将保持零样本性能。

| Model | Config |  AP  | AP50 | AP75  | APS | APM | APL |
| :---- | :----: | :--: | :--: | :---: | :-: | :-: | :-: |
| YOLO-World-v2-L | Zero-shot | 45.7 | 61.6 | 49.8 | 29.9 | 50.0 | 60.8 |
| [YOLO-World-v2-L](./../configs/prompt_tuning_coco/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_prompt_tuning_coco.py) | Prompt tuning | 47.9 | 64.3 | 52.5 | 31.9 | 52.6 | 61.3 | 

完整的配置文件如下所示：

<details><summary>🪐 点击查看完整的提示词微调配置文件</summary>

```python
_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# hyper-parameters
num_classes = 80
num_training_classes = 80
max_epochs = 80  # Maximum training epochs
close_mosaic_epochs = 10
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-3
weight_decay = 0.05
train_batch_size_per_gpu = 16
# 权重下载链接：https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_obj365v1_goldg_cc3mlite_pretrain-ca93cd1f.pth
load_from = 'pretrained_models/yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth'
persistent_workers = False

# model settings
model = dict(type='SimpleYOLOWorldDetector',
             mm_neck=True,
             num_train_classes=num_training_classes,
             num_test_classes=num_classes,
             embedding_path='embeddings/clip_vit_b32_coco_80_embeddings.npy',  # 💡 这里需要替换为我们自己的Text Embedding
             prompt_dim=text_channels,
             num_prompts=80,
             data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
             backbone=dict(_delete_=True,
                           type='MultiModalYOLOBackbone',
                           text_model=None,  # 💡 这里不再使用文本语言模型
                           image_model={{_base_.model.backbone}},
                           frozen_stages=4,
                           with_text_model=False),  # 💡 这里不再使用文本语言模型
             neck=dict(type='YOLOWorldPAFPN',
                       freeze_all=True,  # 💡 冻结Neck的权重（不参与微调）
                       guide_channels=text_channels,
                       embed_channels=neck_embed_channels,
                       num_heads=neck_num_heads,
                       block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
             bbox_head=dict(type='YOLOWorldHead',
                            head_module=dict(
                                type='YOLOWorldHeadModule',
                                freeze_all=True,  # 💡 冻结Head的权重（不参与微调）
                                use_bn_head=True,
                                embed_dims=text_channels,
                                num_classes=num_training_classes)),
             train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# dataset settings
final_transform = [
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction'))
]
mosaic_affine_transform = [
    dict(type='Mosaic',
         img_scale=_base_.img_scale,
         pad_val=114.0,
         pre_transform=_base_.pre_transform),
    dict(type='YOLOv5CopyPaste', prob=_base_.copypaste_prob),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        # img_scale is (width, height)
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114),
        min_area_ratio=_base_.min_area_ratio,
        use_mask_refine=_base_.use_mask2refine)  # 💡 如果我们的数据集没有Segment信息，那么将use_mask_refine=False
]
train_pipeline = [
    *_base_.pre_transform, *mosaic_affine_transform,
    dict(type='YOLOv5MixUp',
         prob=_base_.mixup_prob,
         pre_transform=[*_base_.pre_transform, *mosaic_affine_transform]),
    *_base_.last_transform[:-1], *final_transform
]

train_pipeline_stage2 = [*_base_.train_pipeline_stage2[:-1], *final_transform]

coco_train_dataset = dict(type='YOLOv5CocoDataset',
                          data_root='data/coco',
                          ann_file='annotations/instances_train2017.json',
                          data_prefix=dict(img='train2017/'),
                          filter_cfg=dict(filter_empty_gt=False, min_size=32),
                          pipeline=train_pipeline)

train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)

train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)
test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param'))
]
coco_val_dataset = dict(type='YOLOv5CocoDataset',
                        data_root='data/coco',
                        ann_file='annotations/instances_val2017.json',
                        data_prefix=dict(img='val2017/'),
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),
                        pipeline=test_pipeline)

val_dataloader = dict(dataset=coco_val_dataset)
test_dataloader = val_dataloader
# training settings
default_hooks = dict(param_scheduler=dict(scheduler_type='linear',
                                          lr_factor=0.01,
                                          max_epochs=max_epochs),
                     checkpoint=dict(max_keep_ckpts=-1,
                                     save_best=None,
                                     interval=save_epoch_intervals))
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=5,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])
optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW',
    lr=base_lr,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu),
                     paramwise_cfg=dict(bias_decay_mult=0.0,
                                        norm_decay_mult=0.0,
                                        custom_keys={
                                            'backbone.text_model':
                                            dict(lr_mult=0.01),
                                            'logit_scale':
                                            dict(weight_decay=0.0),
                                            'embeddings':
                                            dict(weight_decay=0.0)
                                        }),
                     constructor='YOLOWv5OptimizerConstructor')

# evaluation settings
val_evaluator = dict(_delete_=True,
                     type='mmdet.CocoMetric',
                     proposal_nums=(100, 1, 10),
                     ann_file='data/coco/annotations/instances_val2017.json',
                     metric='bbox')
find_unused_parameters = True
```

</details>

### 4.2.3 🔥 提示词微调示例（Example of prompt finetuning）

#### 1. 数据集准备

这里还是将数据集转换为COCO格式，详情参考：[普通微调示例之数据集准备](#4.1.6.1)。

#### 2. 生成 text embeddings <a id=4.2.3.2></a>

需要通过 `tools/generate_text_prompts.py` 生成文本嵌入并将其保存为形状为 `NxD`的`numpy.array`。我们首先需要准备一个.json文件，里面是每个类别的captions，这里我们以coco128为例子：

```json
[["person"], ["bicycle"], ["car"], ["motorcycle"], ["airplane"], ["bus"], ["train"], ["truck"], ["boat"], ["traffic light"], ["fire hydrant"], ["stop sign"], ["parking meter"], ["bench"], ["bird"], ["cat"], ["dog"], ["horse"], ["sheep"], ["cow"], ["elephant"], ["bear"], ["zebra"], ["giraffe"], ["backpack"], ["umbrella"], ["handbag"], ["tie"], ["suitcase"], ["frisbee"], ["skis"], ["snowboard"], ["sports ball"], ["kite"], ["baseball bat"], ["baseball glove"], ["skateboard"], ["surfboard"], ["tennis racket"], ["bottle"], ["wine glass"], ["cup"], ["fork"], ["knife"], ["spoon"], ["bowl"], ["banana"], ["apple"], ["sandwich"], ["orange"], ["broccoli"], ["carrot"], ["hot dog"], ["pizza"], ["donut"], ["cake"], ["chair"], ["couch"], ["potted plant"], ["bed"], ["dining table"], ["toilet"], ["tv"], ["laptop"], ["mouse"], ["remote"], ["keyboard"], ["cell phone"], ["microwave"], ["oven"], ["toaster"], ["sink"], ["refrigerator"], ["book"], ["clock"], ["vase"], ["scissors"], ["teddy bear"], ["hair drier"], ["toothbrush"]]
```

我们将这个文件命名为：`data/texts/coco_class_captions.json`。之后我们需要使用CLIP对其进行推理，得到一个embedding向量：

```bash
python tools/generate_text_prompts.py \
    --model openai/clip-vit-base-patch32 \
    --text data/texts/coco128_class_captions.json \
    --out data/texts/coco128_class_captions_embedding.npy
```

> 💡 openai/clip-vit-base-patch32下载地址为：[openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32/tree/main)，将所有文件都下载下来，放到`openai`这个文件夹中即可。

#### 3. 创建和修改配置文件

首先我们需要创建一个配置文件：

```bash
cp configs/prompt_tuning_coco/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_prompt_tuning_coco.py configs/prompt_tuning_coco/yolo_world_v2_s_vlpan_bn_2e-4_80e_1gpus_prompt_tuning_coco128.py
```

之后我们需要修改它的内容，如下所示：

<details><summary>🪐 点击查看完整的配置文件</summary>

```python
_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# hyper-parameters
num_classes = 80  # 💡 修改为自己的类别数
num_training_classes = 80  # 💡 修改为自己的类别数
max_epochs = 80  # 💡 想要微调的epoch数
close_mosaic_epochs = 10
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 16  # 💡 每个GPU的batch大小
# 权重下载链接：https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_obj365v1_goldg_cc3mlite_pretrain-ca93cd1f.pth
load_from = 'pretrained_models/yolo_world_v2_l_obj365v1_goldg_cc3mlite_pretrain-ca93cd1f.pth'
persistent_workers = False

# model settings
model = dict(type='SimpleYOLOWorldDetector',
             mm_neck=True,
             num_train_classes=num_training_classes,
             num_test_classes=num_classes,
             embedding_path='data/texts/coco128_class_captions_embedding.npy',  # 💡 修改为我们自己生成的embedding vector路径
             prompt_dim=text_channels,
             num_prompts=80,  # 💡 也要修改
             freeze_prompt=False,
             data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
             backbone=dict(_delete_=True,
                           type='MultiModalYOLOBackbone',
                           text_model=None,
                           image_model={{_base_.model.backbone}},
                           frozen_stages=4,
                           with_text_model=False),
             neck=dict(type='YOLOWorldPAFPN',
                       freeze_all=True,
                       guide_channels=text_channels,
                       embed_channels=neck_embed_channels,
                       num_heads=neck_num_heads,
                       block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
             bbox_head=dict(type='YOLOWorldHead',
                            head_module=dict(
                                type='YOLOWorldHeadModule',
                                freeze_all=True,
                                use_bn_head=True,
                                embed_dims=text_channels,
                                num_classes=num_training_classes)),
             train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# dataset settings
coco_train_dataset = dict(type='YOLOv5CocoDataset',
                          data_root='data/coco128',
                          ann_file='annotations/train.json',
                          data_prefix=dict(img='train/images/'),
                          filter_cfg=dict(filter_empty_gt=False, min_size=32),
                          pipeline=_base_.train_pipeline)

train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)

coco_val_dataset = dict(type='YOLOv5CocoDataset',
                        data_root='data/coco128',
                        ann_file='annotations/val.json',
                        data_prefix=dict(img='val/images/'),
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),
                        pipeline=_base_.test_pipeline)

val_dataloader = dict(dataset=coco_val_dataset)
test_dataloader = val_dataloader
# training settings
default_hooks = dict(param_scheduler=dict(scheduler_type='linear',
                                          lr_factor=0.01,
                                          max_epochs=max_epochs),
                     checkpoint=dict(max_keep_ckpts=-1,
                                     save_best=None,
                                     interval=save_epoch_intervals))
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=_base_.train_pipeline_stage2)
]
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=5,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])

optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW',
    lr=base_lr,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu),
                     paramwise_cfg=dict(custom_keys={
                                            'backbone.text_model':
                                            dict(lr_mult=0.01),
                                            'logit_scale':
                                            dict(weight_decay=0.0),
                                            'embeddings':
                                            dict(weight_decay=0.0)
                                        }),
                     constructor='YOLOWv5OptimizerConstructor')

# evaluation settings
val_evaluator = dict(_delete_=True,
                     type='mmdet.CocoMetric',
                     proposal_nums=(100, 1, 10),
                     ann_file='data/coco128/annotations/val.json',
                     metric='bbox')
```

</details>


#### 4. 开始训练

```bash
export CUDA_VISIBLE_DEVICES=1
bash tools/dist_train.sh \
    configs/prompt_tuning_coco/yolo_world_v2_s_vlpan_bn_2e-4_80e_1gpus_prompt_tuning_coco128.py \
    1 \
    --amp
```

#### 5. 训练过程展示

```log
2024/06/19 09:21:17 - mmengine - INFO - Load checkpoint from pretrained_models/yolo_world_v2_l_obj365v1_goldg_cc3mlite_pretrain-ca93cd1f.pth
2024/06/19 09:21:17 - mmengine - WARNING - "FileClient" will be deprecated in future. Please use io functions in https://mmengine.readthedocs.io/en/latest/api/fileio.html#file-io
2024/06/19 09:21:17 - mmengine - WARNING - "HardDiskBackend" is the alias of "LocalBackend" and the former will be deprecated in future.
2024/06/19 09:21:17 - mmengine - INFO - Checkpoints will be saved to /home/Le0v1n/code/YOLO-World/work_dirs/yolo_world_v2_l_vlpan_bn_2e-4_80e_1gpus_prompt_tuning_coco128.
2024/06/19 09:21:27 - mmengine - INFO - Epoch(train)  [1][  50/1509]  base_lr: 2.0000e-04 lr: 2.1648e-06  eta: 6:39:41  time: 0.1987  data_time: 0.0376  memory: 14918  grad_norm: nan  loss: 48.7437  loss_cls: 13.6035  loss_bbox: 16.5325  loss_dfl: 18.6077
2024/06/19 09:21:33 - mmengine - INFO - Epoch(train)  [1][ 100/1509]  base_lr: 2.0000e-04 lr: 4.3738e-06  eta: 5:29:57  time: 0.1295  data_time: 0.0203  memory: 4644  grad_norm: 112.0175  loss: 47.6533  loss_cls: 13.0782  loss_bbox: 16.1408  loss_dfl: 18.4343
2024/06/19 09:21:40 - mmengine - INFO - Epoch(train)  [1][ 150/1509]  base_lr: 2.0000e-04 lr: 6.5827e-06  eta: 5:03:34  time: 0.1250  data_time: 0.0161  memory: 4418  grad_norm: 72.2968  loss: 46.9883  loss_cls: 12.8797  loss_bbox: 15.8713  loss_dfl: 18.2373
2024/06/19 09:21:46 - mmengine - INFO - Epoch(train)  [1][ 200/1509]  base_lr: 2.0000e-04 lr: 8.7917e-06  eta: 4:48:44  time: 0.1218  data_time: 0.0121  memory: 4591  grad_norm: 51.6703  loss: 45.8311  loss_cls: 12.2765  loss_bbox: 15.6204  loss_dfl: 17.9342
2024/06/19 09:21:52 - mmengine - INFO - Epoch(train)  [1][ 250/1509]  base_lr: 2.0000e-04 lr: 1.1001e-05  eta: 4:42:53  time: 0.1295  data_time: 0.0219  memory: 4818  grad_norm: 53.6273  loss: 46.9283  loss_cls: 12.7817  loss_bbox: 15.9318  loss_dfl: 18.2147
```

#### 6. FAQ

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏-1：在训练开始的时候，提示很多key匹配不上。
🥳 𝑨𝒏𝒔𝒘𝒆𝒓-1：我也不知道为什么，在官方的issue中发现其他人也有类似的现象。但是经过我自己的训练，这样没有什么大问题，模型仍然可以训练。

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏-2：Prompt Finetuning之后，模型的zero-shot能力消失了。
🥳 𝑨𝒏𝒔𝒘𝒆𝒓-2：这是正常现象，因为我们在训练的过程中使用了text embedding，所以模型中的相关模块的参数被改变了，这就导致了模型丢失zero-shot能力。

## 4.3 重参数化微调（Re-parameterized fine-tuning）

### 4.3.1 原理

重参数化将文本嵌入（text embedding）作为参数合并到模型中。例如，在最终的分类层中，<font color='red'><b>文本嵌入（text embedding）被重参数化为简单的 1×1 卷积层</b></font>。

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-06-17-11-28-09.png
    width=75%></br><center></center>
</div></br>

### 4.3.2 重参数化的主要优势

- **zero-shot**：重参数化后的YOLO-World<font color='green'><b>仍然具有零样本能力</b></font>（Prompt Finetuning之后就没有zero-shot的能力了）！
- **效率**：重参数化的 YOLO-World 具有简单高效的架构，因为 `conv1x1`比`transpose & matmul` 更快。此外，而且还可以进一步优化部署。
- **准确性**：重参数化的YOLO-World支持微调。与普通的 `fine-tuning`或`prompt tuning` 相比，重参数化版本可以独立优化 `neck`和`head`，因为 `neck`和`head` 有不同的参数，<font color='red'><b>⚠️ 不再依赖于 text embeddings</b></font>！例如，在COCO val2017上重参数化微调的YOLO-World的mAP为46.3，而普通微调（Normal Finetuning）版本的mAP为46.1，所有超参数保持不变。

### 4.3.3 如何使用

#### 1. 准备自定义Text Embedding向量

需要通过 `tools/generate_text_prompts.py` 生成文本嵌入并将其保存为形状为 `NxD`的`numpy.array`。

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：不是说Re-parameterize Finetuning不再依赖于Text Embedding了吗？为什么我们还要生成这个`ndarray`的嵌入向量？
🥳 𝑨𝒏𝒔𝒘𝒆𝒓：只是说在训练阶段不再依赖Text Embedding向量，在训练开始前我们是需要它的，目的是生成一个嵌入了该向量的模型。


#### 2. 重参数化预训练权重

这一步需要我们有两个文件：

- text embeddings：文本嵌入
- model checkpoint：模型权重文件

text embeddings我们在第一步刚刚生成，现在是要挑选一个合适的ckpt来进行重参数化。这里我们选择`pretrained_models/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth`进行重参数化。重参数化会改变两个`module`：

- head (`YOLOWorldHeadModule`) 
- neck (`MaxSigmoidCSPLayerWithTwoConv`) 

> 💡 权重下载链接：[yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth)

那我们开始重参数化：

```bash
python tools/reparameterize_yoloworld.py \
    --model pretrained_models/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth \
    --out-dir pretrained_models/re-parameterized/ \
    --text-embed data/texts/coco128_class_captions_embedding.npy \
    --conv-neck
```

然后运行 -> 报错😑：

```
Traceback (most recent call last):
  File "/home/Le0v1n/code/YOLO-World/tools/reparameterize_yoloworld.py", line 139, in <module>
    main()
  File "/home/Le0v1n/code/YOLO-World/tools/reparameterize_yoloworld.py", line 135, in main
    torch.save(model, os.path.join(args.out_dir, model_name))
  File "/root/anaconda3/envs/yolo-world/lib/python3.9/site-packages/torch/serialization.py", line 618, in save
    with _open_zipfile_writer(f) as opened_zipfile:
  File "/root/anaconda3/envs/yolo-world/lib/python3.9/site-packages/torch/serialization.py", line 492, in _open_zipfile_writer
    return container(name_or_buffer)
  File "/root/anaconda3/envs/yolo-world/lib/python3.9/site-packages/torch/serialization.py", line 463, in __init__
    super().__init__(torch._C.PyTorchFileWriter(self.name))
RuntimeError: Parent directory pretrained_models/re-parameterized does not exist.
```

好，那我们自己修改一下这个代码，别让它那么呆：

```python
import argparse
import torch
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser("Reparameterize YOLO-World")
    parser.add_argument('--model', help='model checkpoints to reparameterize')
    parser.add_argument('--out-dir', help='output checkpoints')
    parser.add_argument('--text-embed', help='text embeddings to reparameterized into YOLO-World')
    parser.add_argument('--conv-neck', action='store_true', help='whether using 1x1 conv in RepVL-PAN')

    args = parser.parse_args()
    return args


def convert_head(scale, bias, text_embed):
    N, D = text_embed.shape
    weight = (text_embed * scale.exp()).view(N, D, 1, 1)
    bias = torch.ones(N) * bias
    return weight, bias


def reparameterize_head(state_dict, embeds):

    cls_layers = [
        'bbox_head.head_module.cls_contrasts.0',
        'bbox_head.head_module.cls_contrasts.1',
        'bbox_head.head_module.cls_contrasts.2'
    ]

    for i in range(3):
        scale = state_dict[cls_layers[i] + '.logit_scale']
        bias = state_dict[cls_layers[i] + '.bias']
        weight, bias = convert_head(scale, bias, embeds)
        state_dict[cls_layers[i] + '.conv.weight'] = weight
        state_dict[cls_layers[i] + '.conv.bias'] = bias
        del state_dict[cls_layers[i] + '.bias']
        del state_dict[cls_layers[i] + '.logit_scale']
    return state_dict


def convert_neck_split_conv(input_state_dict, block_name, text_embeds,
                            num_heads):
    if block_name + '.guide_fc.weight' not in input_state_dict:
        return input_state_dict
    guide_fc_weight = input_state_dict[block_name + '.guide_fc.weight']
    guide_fc_bias = input_state_dict[block_name + '.guide_fc.bias']
    guide = text_embeds @ guide_fc_weight.transpose(0,
                                                    1) + guide_fc_bias[None, :]
    N, D = guide.shape
    guide = list(guide.split(D // num_heads, dim=1))
    del input_state_dict[block_name + '.guide_fc.weight']
    del input_state_dict[block_name + '.guide_fc.bias']
    for i in range(num_heads):
        input_state_dict[block_name +
                         f'.guide_convs.{i}.weight'] = guide[i][:, :, None,
                                                                None]
    return input_state_dict


def convert_neck_weight(input_state_dict, block_name, embeds, num_heads):
    guide_fc_weight = input_state_dict[block_name + '.guide_fc.weight']
    guide_fc_bias = input_state_dict[block_name + '.guide_fc.bias']
    guide = embeds @ guide_fc_weight.transpose(0, 1) + guide_fc_bias[None, :]
    N, D = guide.shape
    del input_state_dict[block_name + '.guide_fc.weight']
    del input_state_dict[block_name + '.guide_fc.bias']
    input_state_dict[block_name + '.guide_weight'] = guide.view(
        N, D // num_heads, num_heads)
    return input_state_dict


def reparameterize_neck(state_dict, embeds, type='conv'):

    neck_blocks = [
        'neck.top_down_layers.0.attn_block',
        'neck.top_down_layers.1.attn_block',
        'neck.bottom_up_layers.0.attn_block',
        'neck.bottom_up_layers.1.attn_block'
    ]
    if "neck.top_down_layers.0.attn_block.bias" not in state_dict:
        return state_dict
    for block in neck_blocks:
        num_heads = state_dict[block + '.bias'].shape[0]
        if type == 'conv':
            convert_neck_split_conv(state_dict, block, embeds, num_heads)
        else:
            convert_neck_weight(state_dict, block, embeds, num_heads)
    return state_dict


def main():
    args = parse_args()

    # 加载ckpt
    model = torch.load(args.model, map_location='cpu')
    state_dict = model['state_dict']

    # 加载Text Embedding向量
    embeddings = torch.from_numpy(np.load(args.text_embed))

    # 移除文本编码器
    keys = list(state_dict.keys())
    keys = [x for x in keys if "text_model" not in x]

    state_dict_wo_text = {x: state_dict[x] for x in keys}
    print("✅ Removing text encoder")

    state_dict_wo_text = reparameterize_head(state_dict_wo_text, embeddings)
    print("✅ Reparameterizing HEAD")

    if args.conv_neck:
        neck_type = "conv"
    else:
        neck_type = "linear"

    state_dict_wo_text = reparameterize_neck(state_dict_wo_text, embeddings, neck_type)
    print("✅ Reparameterizing HEAD")

    # 用新内容替换之前的ckpt字典
    model['state_dict'] = state_dict_wo_text
    
    # 保存新的ckpt
    model_name = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # 创建文件夹
    dst_path = out_dir.joinpath(model_name.stem + f'_rep_{neck_type}' + model_name.suffix)
    
    torch.save(model, str(dst_path))
    print(f"✅ The reparameterized ckpt save in {str(dst_path)}.")


if __name__ == "__main__":
    main()
```

代码改动在`main()`函数中，主要是将之前的`os`库替换为了`pathlib`库。

运行完毕后会生成一个新的ckpt：

```log
(yolo-world) root@Xxxxx:/home/Le0v1n/code/YOLO-World# bash tools/re-parameterize_ckpt.sh
✅ Removing text encoder
✅ Reparameterizing HEAD
✅ Reparameterizing HEAD
✅ The reparameterized ckpt save in pretrained_models/re-parameterized/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea_rep_conv.pth.
```

```
pretrained_models
├── re-parameterized
│   └── yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea_rep_conv.pth  # 💡 新生成的ckpt
├── yolo_world_v2_l_obj365v1_goldg_cc3mlite_pretrain-ca93cd1f.pth  # 预训练权重
├── yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth  # 预训练权重
└── yolo_world_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco_ep80-492dc329.pth  # 预训练权重
```

#### 3. 准备模型配置

我们以`configs/finetune_coco/yolo_world_v2_s_rep_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py`这个配置为例进行重参数化训练。在这个配置文件中，主要关注的是：

- `RepYOLOWorldHeadModule`
- `RepConvMaxSigmoidCSPLayerWithTwoConv`

```python
# RepConvMaxSigmoidCSPLayerWithTwoConv
neck=dict(type='YOLOWorldPAFPN',  # 💡 Neck的freeze_all=True没有了，说明neck的参数也会被调整
          guide_channels=num_classes,
          embed_channels=neck_embed_channels,
          num_heads=neck_num_heads,
          block_cfg=dict(type='RepConvMaxSigmoidCSPLayerWithTwoConv',
                         guide_channels=num_classes)),

# RepYOLOWorldHeadModule
bbox_head=dict(head_module=dict(type='RepYOLOWorldHeadModule',  # 💡 Head的freeze_all=True没有了，说明Head的参数也会被调整
                                embed_dims=text_channels,
                                num_guide=num_classes,
                                num_classes=num_classes)),
```

- `neck`和`bbox_head`中的`freeze_all=True`这个参数没有了，说明这两个模块的中的权重会被微调。
- `neck`的`YOLOWorldPAFPN`中的`block_cfg`类型从`MaxSigmoidCSPLayerWithTwoConv`变为了`RepConvMaxSigmoidCSPLayerWithTwoConv`，新增的`Rep`字段表明Reparameterized，即重参数化的。
- `bbox_head`中的`head_module`的类型从`YOLOWorldHeadModule`变为了`RepYOLOWorldHeadModule`，，新增的`Rep`字段表明Reparameterized，即重参数化的。

---

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：之前的Prompt Finetuning的配置文件中有Text Embedding的路径，Re-paramterize Finetuning不需要吗？
🥳 𝑨𝒏𝒔𝒘𝒆𝒓：是的，在Prompt Finetuning中我们需要传入一个Text Embedding向量，从而实现提示词微调。但在重参数化微调中，因为我们在第二步的时候已经使用Text Embedding向量生成了一个ckpt，所以这里只需要传入ckpt路径就行，不需要Text Embedding了。

#### 4. 启动微调

和之前的训练方式一样，我们只是传入的配置文件不同而已。

```bash
./dist_train.sh <配置文件路径> <NUM_GPUS> --amp
```

### 4.3.4 🔥 重参数化微调示例

#### 1. 准备数据集

这里还是将数据集转换为COCO格式，详情参考：[普通微调示例之数据集准备](#4.1.6.1)。

#### 2. 生成 text embeddings

与Prompt Finetuning一样，我们需要生成文本嵌入（Text Embedding），流程参考[4.2.3.2 生成 text embeddings](#4.2.3.2)。

#### 3. 创建和修改配置文件

首先我们需要创建一个配置文件：

```bash
cp configs/finetune_coco/yolo_world_v2_s_rep_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py configs/finetune_coco/yolo_world_v2_s_rep_vlpan_bn_2e-4_80e_1gpus_finetune_coco128.py
```

然后对其进行修改，完整的配置文件示例如下：

<details><summary>🪐 点击查看完整的重参数化微调配置文件</summary>

```python
_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# hyper-parameters
num_classes = 80  # 💡 替换为自己的类别数
num_training_classes = 80  # 💡 替换为自己的类别数
max_epochs = 80  # Maximum training epochs
close_mosaic_epochs = 10
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 16
# ❗ 这里的预训练权重应该选择我们自己生成的
load_from = 'pretrained_models/re-parameterized/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea_rep_conv.pth'
persistent_workers = False
mixup_prob = 0.15
copypaste_prob = 0.3

# model settings
model = dict(type='SimpleYOLOWorldDetector',
             mm_neck=True,
             num_train_classes=num_classes,
             num_test_classes=num_classes,
             reparameterized=True,  # 💡 开启重参数化
             data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
             backbone=dict(_delete_=True,
                           type='MultiModalYOLOBackbone',
                           text_model=None,  # 💡 不再使用语言模型
                           image_model={{_base_.model.backbone}},
                           with_text_model=False),  # 💡 不再使用语言模型
             neck=dict(type='YOLOWorldPAFPN',  # 💡 Neck的freeze_all=True没有了，说明neck的参数也会被调整
                       guide_channels=num_classes,
                       embed_channels=neck_embed_channels,
                       num_heads=neck_num_heads,
                       block_cfg=dict(type='RepConvMaxSigmoidCSPLayerWithTwoConv',  # 💡 block也使用的是Rep开头的，表明是重参数化
                                      guide_channels=num_classes)),
             bbox_head=dict(head_module=dict(type='RepYOLOWorldHeadModule',  # 💡 Head的freeze_all=True没有了，说明Head的参数也会被调整
                                             embed_dims=text_channels,
                                             num_guide=num_classes,
                                             num_classes=num_classes)),
             train_cfg=dict(assigner=dict(num_classes=num_classes)))

# dataset settings
final_transform = [
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction'))
]
mosaic_affine_transform = [
    dict(type='Mosaic',
         img_scale=_base_.img_scale,
         pad_val=114.0,
         pre_transform=_base_.pre_transform),
    dict(type='YOLOv5CopyPaste', prob=copypaste_prob),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        # img_scale is (width, height)
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114),
        min_area_ratio=_base_.min_area_ratio,
        use_mask_refine=_base_.use_mask2refine)
]
train_pipeline = [
    *_base_.pre_transform, *mosaic_affine_transform,
    dict(type='YOLOv5MixUp',
         prob=mixup_prob,
         pre_transform=[*_base_.pre_transform, *mosaic_affine_transform]),
    *_base_.last_transform[:-1], *final_transform
]

train_pipeline_stage2 = [*_base_.train_pipeline_stage2[:-1], *final_transform]

coco_train_dataset = dict(type='YOLOv5CocoDataset',
                          data_root='data/coco128',
                          ann_file='annotations/train.json',
                          data_prefix=dict(img='train/images/'),
                          filter_cfg=dict(filter_empty_gt=False, min_size=32),
                          pipeline=train_pipeline)

train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)

train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)
test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param'))
]
coco_val_dataset = dict(type='YOLOv5CocoDataset',
                        data_root='data/coco128',
                        ann_file='annotations/val.json',
                        data_prefix=dict(img='val/images/'),
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),
                        pipeline=test_pipeline)

val_dataloader = dict(dataset=coco_val_dataset)
test_dataloader = val_dataloader
# training settings
default_hooks = dict(param_scheduler=dict(scheduler_type='linear',
                                          lr_factor=0.01,
                                          max_epochs=max_epochs),
                     checkpoint=dict(max_keep_ckpts=-1,
                                     save_best='auto',  # 💡 从None修改为'auto'
                                     interval=save_epoch_intervals))
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=5,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])
optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW',
    lr=base_lr,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu),
                     constructor='YOLOWv5OptimizerConstructor')

# evaluation settings
val_evaluator = dict(_delete_=True,
                     type='mmdet.CocoMetric',
                     proposal_nums=(100, 1, 10),
                     ann_file='data/coco128/annotations/val.json',
                     metric='bbox')
```

</details>

#### 4. 开始训练

```bash
export CUDA_VISIBLE_DEVICES=1
bash tools/dist_train.sh \
    configs/finetune_coco/yolo_world_v2_s_rep_vlpan_bn_2e-4_80e_1gpus_finetune_coco128.py \
    1 \
    --amp
```

#### 5. 训练过程展示

```
2024/06/20 06:22:44 - mmengine - INFO - Load checkpoint from pretrained_models/re-parameterized/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea_rep_conv.pth
2024/06/20 06:22:44 - mmengine - WARNING - "FileClient" will be deprecated in future. Please use io functions in https://mmengine.readthedocs.io/en/latest/api/fileio.html#file-io
2024/06/20 06:22:44 - mmengine - WARNING - "HardDiskBackend" is the alias of "LocalBackend" and the former will be deprecated in future.
2024/06/20 06:22:44 - mmengine - INFO - Checkpoints will be saved to /home/Le0v1n/code/YOLO-World/work_dirs/yolo_world_v2_s_rep_vlpan_bn_2e-4_80e_1gpus_finetune_pedestrian30k.
2024/06/20 06:22:56 - mmengine - INFO - Epoch(train)  [1][  50/1509]  lr: 2.1648e-06  eta: 7:48:04  time: 0.2327  data_time: 0.0576  memory: 8696  grad_norm: nan  loss: 59.4169  loss_cls: 20.9526  loss_bbox: 18.8704  loss_dfl: 19.5938
2024/06/20 06:23:03 - mmengine - INFO - Epoch(train)  [1][ 100/1509]  lr: 4.3738e-06  eta: 6:04:49  time: 0.1302  data_time: 0.0266  memory: 4375  grad_norm: 496.5040  loss: 52.9109  loss_cls: 16.6957  loss_bbox: 17.5310  loss_dfl: 18.6842
2024/06/20 06:23:09 - mmengine - INFO - Epoch(train)  [1][ 150/1509]  lr: 6.5827e-06  eta: 5:31:30  time: 0.1320  data_time: 0.0365  memory: 4389  grad_norm: 477.9625  loss: 50.6395  loss_cls: 14.8803  loss_bbox: 17.0048  loss_dfl: 18.7545
2024/06/20 06:23:16 - mmengine - INFO - Epoch(train)  [1][ 200/1509]  lr: 8.7917e-06  eta: 5:15:02  time: 0.1324  data_time: 0.0474  memory: 4389  grad_norm: 475.2882  loss: 51.2433  loss_cls: 14.6030  loss_bbox: 17.6751  loss_dfl: 18.9652
2024/06/20 06:23:23 - mmengine - INFO - Epoch(train)  [1][ 250/1509]  lr: 1.1001e-05  eta: 5:05:49  time: 0.1342  data_time: 0.0332  memory: 4975  grad_norm: 417.9657  loss: 50.3242  loss_cls: 14.2822  loss_bbox: 17.4215  loss_dfl: 18.6204
2024/06/20 06:23:29 - mmengine - INFO - Epoch(train)  [1][ 300/1509]  lr: 1.3210e-05  eta: 4:58:14  time: 0.1300  data_time: 0.0280  memory: 4335  grad_norm: 440.3105  loss: 48.2986  loss_cls: 13.4765  loss_bbox: 16.5409  loss_dfl: 18.2812
2024/06/20 06:23:35 - mmengine - INFO - Epoch(train)  [1][ 350/1509]  lr: 1.5419e-05  eta: 4:51:48  time: 0.1265  data_time: 0.0245  memory: 4455  grad_norm: 415.5017  loss: 47.7501  loss_cls: 13.3075  loss_bbox: 16.3856  loss_dfl: 18.0570
2024/06/20 06:23:41 - mmengine - INFO - Epoch(train)  [1][ 400/1509]  lr: 1.7628e-05  eta: 4:45:45  time: 0.1218  data_time: 0.0200  memory: 4469  grad_norm: 459.6522  loss: 48.0474  loss_cls: 13.4490  loss_bbox: 16.4455  loss_dfl: 18.1529
```

#### 6. FAQ

暂无。

# 5. 前置知识

## 5.1 零样本（zero-shot）<a id=6.1></a>

### 5.1.1 概念

零样本学习（zero-shot learning）是机器学习领域中的一种技术，它允许模型在没有接受过<font color='red'><b>特定类别</b></font>训练数据的情况下，识别或预测这些类别。这通常通过利用模型对其他类别的已有知识来实现，或者通过某种形式的语义或属性描述来辅助模型理解新的类别。

> 💡 推荐阅读《[零次学习（Zero-Shot Learning）入门](https://zhuanlan.zhihu.com/p/34656727)》，该文章讲得非常好。

### 5.1.2 FAQ

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：zero-shot和测试集有什么区别，不都是模型没有见过的吗？
🥳 𝑨𝒏𝒔𝒘𝒆𝒓：确实，zero-shot learning 和测试集都涉及到模型在面对未见过的数据时的表现，但它们之间存在一些关键的区别：

- 测试集：模型在训练过程中没有见到过该图片，让模型去预测，模型会预测出它<font color='red'><b>见过的类别</b></font>（注意这里是类别）。
- zero-shot：模型在训练过程中没有见到过该类别，让模型去预测，模型不光会预测出它见过的类别，也会<font color='red'><b>预测出它没有见过的类别</b></font>。

---

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：zero-shot、one-shot、few-shot有什么区别？
🥳 𝑨𝒏𝒔𝒘𝒆𝒓：

|           | 中文翻译 | 特点                                                                                                                 |
| :-------: | :------: | :------------------------------------------------------------------------------------------------------------------- |
| zero-shot |  零样本  | 模型从来没有学习过这个类别的图片，但仍然可以识别出这个类别                                                           |
| one-shot  |  单样本  | 模型在训练过程中只学习过<font color='red'><b>一张</b></font>有该类别的图片，在后续的使用中，模型可以正确推理出该类别 |
| few-shot  |  少样本  | 模型在训练过程中只学习过<font color='red'><b>少量</b></font>有该类别的图片，在后续的使用中，模型可以正确推理出该类别 |

- 三者的简单示例：
  - **zero-shot**：如果一个模型在训练时学会了识别马、熊猫和鸟，它可以在没有见过的类别（如斑马）上进行预测，因为它了解到斑马是一种动物，拥有和马一样的体型，有类似熊猫的黑白色毛。zero-shot能够通过学习任务之间的关系和共享的特征来实现对未知任务的处理。
  - **one-shot**：如果一个模型只见过一张猫的图片，它可以通过这张图片进行学习，并在之后对新的猫的图像进行分类。one-shot是在非常有限的数据情况下进行学习和推断的一种能力。
  - **few-shot**：如果一个模型仅仅通过观察几个图像，就可以学会识别不同品种的狗，然后能够对新的狗图像进行分类。few-shot要求模型能够从少量示例中抽取出关键特征和模式，以便进行准确的预测。

## 5.2 CLIP（Contrastive Language-Image pre-training）<a id=6.2></a>

### 5.2.1 介绍

CLIP（Contrastive Language–Image Pre-training）是一种多模态学习模型，由OpenAI在2021年提出。它的作用主要包括：

1. **图像和文本的联合表示学习**：CLIP通过在大量图像和文本对上进行预训练，学习<font color='blue'><b>将图像内容与相应的描述文本映射到一个共同的特征空间中</b></font>。
2. **零样本（zero-shot）分类**：CLIP能够在没有传统训练过程的情况下，对图像进行分类。<font color='green'><b>只需要提供类别的文本描述，CLIP就可以识别图像中的对象，即使它之前没有见过这些具体类别</b></font>。
3. **图像检索**：利用CLIP模型，可以通过文本查询来检索与文本描述最匹配的图像。
4. **文本到图像的生成**：虽然CLIP本身不是一个生成模型，但它的编码器（Encoder）可以与生成模型（如GANs）结合，用于生成与文本描述相匹配的图像。
5. **跨模态对比学习**：CLIP通过最小化正样本对（pair）之间的距离并最大化负样本对（pair）之间的距离来进行训练，从而<font color='pink'><b>学习图像和文本之间的相关性</b></font>。
6. **多语言支持**：CLIP支持多种语言的文本输入，这使得它能够处理不同语言的图像描述。

> CLIP模型因其强大的通用性和灵活性，在图像和文本的多模态任务中被广泛研究和应用。

### 5.2.2 输入输出

CLIP模型的输入如下：

1. **图像输入**：CLIP模型接受图像作为输入。这些图像可以是JPEG、PNG等格式，它们首先被模型预处理，包括调整大小、归一化等步骤，以适应模型的输入要求。
2. **文本输入**：CLIP模型同时接受文本描述作为输入。文本可以是类别名称、物体描述、场景描述等自然语言描述。文本输入通常经过分词处理，并转换为模型能理解的嵌入表示。

CLIP模型的输出如下：

1. **图像-文本嵌入**：CLIP模型的编码器会将输入的图像和文本转换为高维空间中的嵌入向量。这些嵌入向量捕捉了图像内容和文本描述的语义信息。
2. **相似度分数**：<font color='red'><b>对于给定的图像和文本对，CLIP模型输出一个相似度分数，该分数表示图像与文本描述的匹配程度</b></font>。分数越高，表示模型认为图像和文本越相关。
3. **分类结果**（零样本分类）：如果文本输入是类别描述，CLIP可以输出图像属于各个类别的概率分布，从而实现零样本分类。
4. **检索结果**：在图像检索任务中，CLIP可以为给定的文本查询返回最相关的图像集合。、
5. **特征图**（高级应用）：在某些应用中，CLIP的中间层可以提供图像的特征图，这些特征图可以用于更复杂的视觉分析任务。

> CLIP模型的设计使其能够处理多种模态的输入，并在不同的任务中提供有用的输出，这使得它在多模态学习和人工智能领域非常受欢迎。

### 5.2.3 FAQ

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：CLIP可以只输入一个文本或一张图片吗？
🥳 𝑨𝒏𝒔𝒘𝒆𝒓：CLIP模型设计时主要是为了处理成对（pair）的输入，即图像和文本的组合，以便学习两者之间的关联。然而，模型的两个主要组件——图像编码器（Image Encoder）和文本编码器（Text Encoder）都是可以独立使用的。

1. **文本编码器（Text Encoder）**：可以单独接收文本输入，并将其编码成嵌入向量（embedding vector）。这个向量可以代表文本的语义内容，用于各种下游任务，比如文本相似度计算、文本聚类等。

2. **图像编码器（Image Encoder）**：同样可以独立接收图像输入，并将其编码成嵌入向量（embedding vector）。这些向量可以用于图像检索、图像相似度计算等任务。

尽管CLIP的完整模型是为图像-文本对设计的，但如果我们只需要处理文本或者图像中的一个，可以只使用相应的编码器部分。例如，如果我们有一个文本，我们想知道它与其他文本的相似度，我们可以使用文本编码器来获取嵌入向量，然后比较这些向量。

然而，如果我们想要CLIP模型输出图像与文本的匹配分数，那么就需要同时提供图像和文本的输入。如果只有文本而没有图像，就无法使用CLIP模型来评估图像与文本的一致性或进行零样本分类等任务。

## 5.3 开集目标检测<a id=6.3></a>

> 推荐阅读文章《[Grounding DINO检测一切](https://zhuanlan.zhihu.com/p/664623532)》。

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：zero-shot和开集目标检测有什么关系？
🥳 𝑨𝒏𝒔𝒘𝒆𝒓：Zero-shot learning（零样本学习）和开集目标检测（Open-set Object Detection）是机器学习领域中的两种不同的概念，但它们在某些方面存在联系：

1. **Zero-shot Learning**：
   - 零样本学习的目标是让模型能够在没有直接训练数据的情况下识别新的类别。这意味着模型需要在训练阶段学习到足够的泛化能力，以便在面对未知类别时做出正确的预测。
   - 零样本学习<font color='red'><b>通常依赖于辅助信息，如类别的描述、属性或相似类别的语义信息</b></font>。
2. **开集目标检测（Open-set Object Detection）**：
   - 开集目标检测是指模型在检测过程中可能遇到训练阶段未见过的新类别。与零样本学习不同，开集目标检测并不要求模型能够识别这些新类别，而是要求模型能够识别出这些未知类别的存在，即使它不能准确地分类它们。
   - 开集目标检测的挑战在于区分已知类别和未知类别，并且对未知类别做出适当的响应。

**两者的关系**：
- **泛化能力**：零样本学习和开集目标检测都要求模型具备良好的泛化能力。零样本学习侧重于利用泛化能力识别新类别，而开集目标检测侧重于利用泛化能力区分已知和未知类别。
- **未知类别**：在零样本学习中，模型需要识别完全未知的类别；而在开集目标检测中，模型需要识别并区分出未知类别，但不一定需要对它们进行分类。
- **辅助信息的使用**：零样本学习通常依赖于辅助信息，如类别描述，来帮助识别新类别。开集目标检测可能不依赖于这些辅助信息，而是依赖于模型的泛化能力和对数据分布的理解。
- **应用场景**：零样本学习和开集目标检测都可以应用于现实世界中，其中模型可能遇到未在训练数据中见过的新对象或类别。
- **挑战和限制**：两者都面临着模型泛化能力的挑战，以及如何处理和识别未知类别的限制。

总的来说，零样本学习和开集目标检测都涉及到模型对未知类别的处理，但它们的焦点和目标略有不同。零样本学习更侧重于识别新类别，而开集目标检测更侧重于区分已知和未知类别。尽管如此，两者在提高模型泛化能力和处理未知类别方面存在一定的联系。

## 5.4 LVIS数据集<a id=6.4></a>

LVIS（Large Vocabulary Instance Segmentation）数据集是由Facebook AI Research (FAIR)开发并发布的一个大规模细粒度词汇级标记数据集。这个数据集专门用于对象检测和实例分割的研究基准，它包含了超过1000类物体的约200万个高质量的实例分割标注，涵盖了164k大小的图像。

**LVIS数据集的特点包括**：
1. **大规模和细粒度**：数据集覆盖了广泛的物体类别，提供了详尽的标注，包括小的、部分被遮挡的或难以辨认的对象实例。
2. **高质量标注**：与COCO和 ADE20K数据集相比，LVIS数据集的标注质量更高，具有更大的重叠面积和更好的边界连续性。
3. **长尾分布**：LVIS数据集反映了自然图像中类别的Zipfian分布，即<font color='red'><b>少数常见类别和大量罕见类别的长尾分布</b></font>。
4. **评估优先的设计原则**：数据集的构建采用了评估优先的设计原则，即首先确定如何执行定量评估，然后设计和构建数据集收集流程以满足评估所需数据的需求。
5. **联合数据集**：LVIS由大量较小的组成数据集联合形成，每个小数据集为单个类别提供详尽标注的基本保证，即该类别的所有实例都被标注。这种设计减少了整体的标注工作量，同时保持了评估的公平性。

LVIS数据集的构建过程包括六个阶段：目标定位、穷尽标记、实例分割、验证、穷尽标注验证以及负例集标注。数据集的词汇表 V是通过迭代过程构建的，从大型超级词汇表开始，并使用目标定位过程逐步缩小，<font color='red'><b>最终确定包含 1723个同义词的词汇表</b></font>，这也是可以出现在 LVIS中的类别数量的上限。

### 5.4.1 🔥 COCO数据集

#### 1. 介绍

COCO2017数据集，全称为Common Objects in Context 2017，是一个大型的、丰富且具有挑战性的对象检测、分割和字幕生成数据集。它是Common Objects in Context（COCO）数据集系列中的一个版本，由微软和哈佛的研究人员创建，并在2017年发布。

以下是COCO2017数据集的一些关键特点：

1. **多任务数据集**：COCO2017不仅包含对象检测任务（detect），还包括对象分割（Segment）和字幕生成任务（caption）。
2. **大规模**：数据集包含超过330,000张标记图像，涵盖了80个类别的对象，包括日常物品、动物、车辆等。
3. **高质量标注**：图像中的每个对象都有详细的标注，包括<font color='red'><b>边界框、分割掩码和/或字幕</b></font>。
4. **类别丰富**：数据集中的对象类别非常丰富，包括人、车辆、动物、家具、电子产品等。
5. **上下文信息**：COCO数据集的一个特点是强调对象的上下文信息，即对象与其周围环境的关系。
6. **挑战性**：由于图像中对象的多样性和复杂性，COCO数据集对计算机视觉算法提出了很高的挑战。
7. **广泛使用**：COCO数据集在计算机视觉领域被广泛使用，是许多算法基准测试的标准数据集。
8. **年度竞赛**：COCO数据集还与年度的COCO竞赛相关联，该竞赛吸引了全球的研究团队参与，推动了计算机视觉技术的发展。

COCO2017数据集通常分为三个部分：训练集（Training set）、验证集（Validation set）和测试集（Test set）。训练集用于模型的训练，验证集用于模型的调优和验证，而测试集则用于最终评估模型的性能。数据集的组织结构和详细的标注信息使其成为研究和开发先进视觉算法的重要资源。

#### 2. 目录结构

COCO2017数据集的目录结构组织得非常清晰，便于管理和使用数据。以下是目录结构：

```
coco2017
├── annotations
│   ├── captions_train2017.json
│   ├── captions_val2017.json
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── test2017
│   ├── 000000000001.jpg
│   ├── 000000000016.jpg
│   └── ...
├── train2017
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ...
└── val2017
    ├── 000000000139.jpg
    ├── 000000000285.jpg
    └── ...
```


- `coco2017`：这是数据集的根目录，包含了所有相关的子目录和文件。
  - `annotations`：这个目录包含了所有与注释相关的JSON文件，用于存储图像中对象的标注信息，包括字幕、实例分割和人体关键点。
    - `captions_train2017.json`和`captions_val2017.json`：这两个文件包含了训练集和验证集图像的字幕信息。
    - `instances_train2017.json`和`instances_val2017.json`：这两个文件包含了训练集和验证集图像中对象的实例分割信息，即每个对象的精确像素级掩码。
    - `person_keypoints_train2017.json`和`person_keypoints_val2017.json`：这两个文件专门包含了训练集和验证集中人体图像的关键点标注信息。
  - `test2017`：这个目录包含了测试集的图像文件。测试集的图像用于算法的最终评估，通常不包含标注信息，或者标注信息是隐藏的，仅用于官方评估。
    - 目录内包含图像文件，文件名以`.jpg`结尾，文件名前缀是连续的数字，表示图像的唯一标识符。
  - `train2017`：这个目录包含了训练集的图像文件，用于模型的训练。
    - 同`test2017`，目录内包含以数字命名的`.jpg`图像文件。
  - `val2017`：这个目录包含了验证集的图像文件，用于模型的评估和调参。
    - 同`test2017`和`train2017`，目录内包含以数字命名的`.jpg`图像文件。

整个目录结构将图像数据和注释数据清晰地分开，便于在不同的任务（如训练、验证和测试）中使用。此外，通过将训练集、验证集和测试集分别存放在不同的目录中，可以方便地进行模型的训练和评估。

#### 3. 目标检测标签文件

<details><summary>🪐 点击查看COCO标签内容</summary>

```json
{
    "info": {
        "description": "COCO 2017 Dataset",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2017,
        "contributor": "COCO Consortium",
        "date_created": "2017/09/01"
    },
    "licenses": [
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
        },
        {
            "url": "http://creativecommons.org/licenses/by-nc/2.0/",
            "id": 2,
            "name": "Attribution-NonCommercial License"
        },
        {
            "...": "..."
        }
    ],
    "images": [
        {
            "license": 4,
            "file_name": "000000397133.jpg",
            "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
            "height": 427,
            "width": 640,
            "date_captured": "2013-11-14 17:02:52",
            "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
            "id": 397133
        },
        {
            "license": 1,
            "file_name": "000000037777.jpg",
            "coco_url": "http://images.cocodataset.org/val2017/000000037777.jpg",
            "height": 230,
            "width": 352,
            "date_captured": "2013-11-14 20:55:31",
            "flickr_url": "http://farm9.staticflickr.com/8429/7839199426_f6d48aa585_z.jpg",
            "id": 37777
        },
        {
            "...": "..."
        }
    ],
    "annotations": [
        {
            "segmentation": [
                [
                    510.66,
                    423.01,
                    511.72,
                    "......",
                    424.6,
                    498.02,
                    510.45,
                    423.01
                ]
            ],
            "area": 702.1057499999998,
            "iscrowd": 0,
            "image_id": 289343,
            "bbox": [
                473.07,
                395.93,
                38.65,
                28.67
            ],
            "category_id": 18,
            "id": 1768
        },
        {
            "segmentation": [
                [
                    289.74,
                    443.39,
                    302.29,
                    "......", 
                    288.64,
                    444.27,
                    291.88,
                    443.74
                ]
            ],
            "area": 27718.476299999995,
            "iscrowd": 0,
            "image_id": 61471,
            "bbox": [
                272.1,
                200.23,
                151.97,
                279.77
            ],
            "category_id": 18,
            "id": 1773
        },
        {
            "...": "..."
        }
    ],
    "categories": [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person"
        },
        {
            "supercategory": "vehicle",
            "id": 2,
            "name": "bicycle"
        },
        {
            "...": "..."
        }
    ]
}
```

</details>

1. **info（信息）**:
   - 描述（description）: COCO 2017数据集
   - 网址（url）: [COCO Dataset](http://cocodataset.org)
   - 版本（version）: 1.0
   - 年份（year）: 2017
   - 贡献者（contributor）: COCO Consortium
   - 创建日期（date_created）: 2017年9月1日
2. **licenses（许可协议）**:
   - 包含多个许可协议对象，每个对象有以下属性：
     - url（网址）: 指向许可协议的链接
     - id（编号）: 许可协议的唯一标识符
     - name（名称）: 许可协议的名称
3. **images（图像）**:
   - 包含多个图像对象，每个对象有以下属性：
     - license（许可）: 图像使用的许可协议编号
     - file_name（文件名）: 图像文件的名称
     - coco_url（COCO网址）: COCO数据集中图像的链接
     - height（高度）: 图像的高度（像素）
     - width（宽度）: 图像的宽度（像素）
     - date_captured（拍摄日期）: 图像拍摄的日期和时间
     - flickr_url（Flickr网址）: 图像在Flickr上的链接
     - id（编号）: 图像的唯一标识符
4. **annotations（注释）**:
   - 包含多个注释对象，每个对象有以下属性：
     - segmentation（分割）: 图像中对象的多边形顶点坐标列表
     - area（面积）: 多边形所围成的区域面积
     - iscrowd（是否为人群）: 一个标志，表示该注释是否表示一个人群
     - image_id（图像编号）: 注释所对应的图像的唯一标识符
     - bbox（边界框）: 表示对象在图像中的位置和大小的边界框（格式为[x_min, y_min, width, height]）
     - category_id（类别编号）: 注释所属的类别编号
     - id（编号）: 注释的唯一标识符
5. **categories（类别）**:
   - 包含多个类别对象，每个对象有以下属性：
     - supercategory（上级类别）: 类别的上级分类
     - id（编号）: 类别的唯一标识符
     - name（名称）: 类别的名称

#### 4. captions标签文件

<details><summary>🪐 点击查看captions.json的内容</summary>

```json
{
    "info": {
        "description": "COCO 2017 Dataset",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2017,
        "contributor": "COCO Consortium",
        "date_created": "2017/09/01"
    },
    "licenses": [
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
        },
        {
            "...": "..."
        },
    ],
    "images": [
        {
            "license": 4,
            "file_name": "000000397133.jpg",
            "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
            "height": 427,
            "width": 640,
            "date_captured": "2013-11-14 17:02:52",
            "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
            "id": 397133
        },
        {
            "license": 1,
            "file_name": "000000037777.jpg",
            "coco_url": "http://images.cocodataset.org/val2017/000000037777.jpg",
            "height": 230,
            "width": 352,
            "date_captured": "2013-11-14 20:55:31",
            "flickr_url": "http://farm9.staticflickr.com/8429/7839199426_f6d48aa585_z.jpg",
            "id": 37777
        },
        {
            "...": "..."
        }
    ],
    "annotations": [
        {
            "image_id": 179765,
            "id": 38,
            "caption": "A black Honda motorcycle parked in front of a garage."
        },
        {
            "image_id": 179765,
            "id": 182,
            "caption": "A Honda motorcycle parked in a grass driveway"
        },
        {
            "image_id": 190236,
            "id": 401,
            "caption": "An office cubicle with four different types of computers."
        },
        {
            "image_id": 331352,
            "id": 441,
            "caption": "A small closed toilet in a cramped space."
        },
        {
            "...": "..."
        }
    ]
}
```

</details>

1. **info（信息）**：和前面的一样，这里不再赘述。
2. **licenses（许可协议）**：和前面的一样，这里不再赘述。
3. **images（图像）**：和前面的一样，这里不再赘述。
4. **annotations（注释）**:
   - 包含字幕的列表，每个字幕是一个对象，具有以下属性：
     - `image_id`（图像编号）: 与字幕关联的图像的唯一标识符。
     - `id`（编号）: 注释的唯一标识符。
     - `caption`（字幕）: 图像的描述性文本，用自然语言描述图像内容。

例如，注释中的一条记录：

```json
{
    "image_id": 179765,
    "id": 38,
    "caption": "A black Honda motorcycle parked in front of a garage."
}
```

我们从COCO官网获取这张图片：

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-06-18-11-34-02.png
    width=35%></br><center>http://images.cocodataset.org/val2017/000000179765.jpg</center>
</div></br>

表示图像ID为179765的图像有一个字幕，该字幕的ID是38，描述是"A black Honda motorcycle parked in front of a garage."（一辆黑色本田摩托车停在车库前）。

我们也发现，还有一条注释也对这张图片进行了captions：

```json
{
    "image_id": 179765,
    "id": 182,
    "caption": "A Honda motorcycle parked in a grass driveway"
}
```

这里可以发现，<font color='red'><b>一张图片不一定只有一个caption，有可能会有多个captions</b></font>。

### 5.4.4 MixedGrounding数据集

和传统的目标检测数据集相比，MixedGrounding数据集多了文字描述，即<font color='red'><b>一张图片有一个caption</b></font>。

# 参考文献

1. [YOLO-World/docs](https://github.com/AILab-CVC/YOLO-World/tree/master/docs)
2. [Zero Shot、One Shot、Few Shot的通俗理解](https://blog.51cto.com/u_15408171/7004231)