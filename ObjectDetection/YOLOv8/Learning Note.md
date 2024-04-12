# 1. 概述

## 1.1 项目介绍

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 是一种前沿、最先进 (SOTA) 的模型，它在之前的 YOLO 版本的成功基础上引入了新功能和改进，以提高性能和灵活性。YOLOv8 旨在快速、准确、易于使用，是广泛应用于目标检测和跟踪、实例分割、图像分类和姿态估计等任务的优秀选择。

与之前的 YOLOv5 不同的是，YOLOv8 是在一个名为 Ultralytics 项目下，该项目将该团队之前制作的 YOLOv3、YOLOv5整合到了一起，并添加了 YOLOv8。初次之外，Ultralytics 更是整合了 YOLOv6、YOLOv9。

## 1.2 安装项目

我们有两种方式使用 Ultralytics 这个项目，第一种方式是我们就是用来训练模型，不修改具体的代码；第二种方式则是我们会修改代码。

> 这里我推荐大家使用第二种方法，适用方法更加广泛。

### 1.2.1 第一种方法

我们就使用内置的代码来训练、预测、评估模型，不会对模型进行修改，那么我们就可以直接通过安装 `ultralytics` 这个库，那么这样会导致项目中名为 `ultralytics` 的库不会生效了。

```bash
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
```

通过这种方式安装 `ultralytics` 库之后，原本项目中名为 `ultralytics` 的文件夹就不会生效了。所以当我们修改代码后并不会生效，因为我们用的就不是项目中的 `ultralytics` 文件夹。

### 1.2.2 🌟 第二种方法

这里推荐使用第二种方式，我们直接项目中的 `ultralytics` 这个文件夹当做一个包。安装命令为：

```bash
pip install -v -e .
```

运行完上面的命令后，我们使用 `pip list` 查看已安装的库，可以发现 `ultralytics` 这个库已经安装完毕了，并且后面有一个地址，这个地址其实就是我们的项目。意思就是说，上面的命令将我们本地的项目封装为一个 Python 的库，所以我们修改本地的代码，也是可以生效的。

> 具体生效的逻辑是因为我们添加了 `-e` 这个参数，`-e: editable`，表示可编辑的。

## 1.3 快速上手

在我们安装好 ultralytics 库并下载 ultralytics 项目后，可以直接使用命令行（Command Line Interface, CLI）进行快速推理一张图片、视频、视频流、摄像头等等：

```bash
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

除了使用命令行来进行推理外，我们也可以写一个 Python 脚本来进行相同的操作：

```python
from ultralytics import YOLO


# ============= 加载模型 =============
# 方法1：通过yaml文件新建一个模型(根据yaml文件中的模型定义自动搭建一个模型)
model = YOLO('yolov8n.yaml')  

# 方法2：加载一个训练好的模型（直接从pt文件中读取模型架构从而搭建模型）
model = YOLO('yolov8n.pt')  

# ============= 模型训练 =============
model.train(data='coco128.yaml', epochs=3)  # 训练 coco128.yaml 中定义的数据集，并且 epochs 为 3
metrics = model.val()  # 让模型在验证集中测试模型性能
 
# ============= 模型训练完毕后做的事情 =============
results = model("https://ultralytics.com/images/bus.jpg")  # 让模型推理一张 web 图片
path = model.export(format='onnx')  # 模型转换为 onnx 格式（不影响原来的 .pt 文件）
```

## 1.3 YOLOv8 支持的任务

YOLOv8 的团队不光提供了目标检测的模型，还基于 YOLOv8 开发了其他模型，下面我们对其进行简单的介绍。为了增加模型的适用范围，官方提供了不同规格的模型，其含义分别如下：

|规格|含义|示例|
|:-|:-|:-|
|YOLOv8 Nano     |非常小|YOLOv8n|
|YOLOv8 Small    |小|YOLOv8s|
|YOLOv8 Medium   |中|YOLOv8m|
|YOLOv8 Large    |大|YOLOv8l|
|YOLOv8 Extra Large|非常大|YOLOv8x|

### 1.3.1 目标检测模型

|模型名称|输入图片大小|mAP@50-95|CPU@ONNX Speed (ms)|A100@TensorRT (ms)|params (M)|FLOPs (B)|
|:-|:-|:-|:-|:-|:-|:-|
|YOLOv8n|640|37.3|80.4|0.99|3.2|8.7|
|YOLOv8s|640|44.9|128.4|1.20|11.2|28.6|
|YOLOv8m|640|50.2|234.7|1.83|25.9|78.9|
|YOLOv8l|640|52.9|375.2|2.39|43.7|165.2|
|YOLOv8x|640|53.9|479.1|3.53|68.2|257.8|

### 1.3.2 分割模型

实例分割比物体检测更进一步，涉及在图像中识别并将各个物体从图像的其余部分进行分割。

实例分割模型的输出是一组掩膜或轮廓，用于勾勒图像中的每个物体，同时还包括每个物体的类别标签和置信度分数。实例分割在我们需要知道物体在图像中的位置以及它们的确切形状时非常有用。

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-04-12-11-44-10.png
    width=60%>
    <center></center>
</div></br>

|模型名称|输入图片大小|mAP<sup>box<br>50-95|mAP<sup>mask<br>50-95|Speed<br><sup>CPU ONNX<br>(ms)|Speed<br><sup>A100 TensorRT<br>(ms)|params<br><sup>(M)|FLOPs<br><sup>(B)|
|:-|:-|:-|:-|:-|:-|:-|:-|
|YOLOv8n-seg|640|36.7|30.5|96.1|1.21|3.4|12.6|
|YOLOv8s-seg|640|44.6|36.8|155.7|1.47|11.8|42.6|
|YOLOv8m-seg|640|49.9|40.8|317.0|2.18|27.3|110.2|
|YOLOv8l-seg|640|52.3|42.6|572.4|2.79|46.0|220.5|
|YOLOv8x-seg|640|53.4|43.4|712.1|4.02|71.8|344.1|

> 💡  需要注意的是，这里的指标仍然是目标检测中使用的 mAP 而非 mIoU。很多人在 Issue 中提出了添加 mIoU，但官方表示不会加入 🤣

### 1.3.3 姿态估计模型

姿态估计是一项任务，涉及在图像中确定特定点的位置，通常称为关键点。关键点可以表示对象的各个部分，如关节、标志性或其他独特的特征。关键点的位置通常表示为一组 2D `[x，y]` 或 3D `[x，y，visible]` 坐标。

姿态估计模型的输出是一组代表图像中对象上关键点的点，通常**还包括每个点的置信度分数**。当我们需要识别场景中对象的特定部分以及它们相对位置时，姿态估计是一个很好的选择。

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-04-12-11-42-31.png
    width=80%>
    <center>基于 YOLOv8 的人体姿态估计示例</center>
</div></br>

下面是基于 YOLOv8 的姿态估计模型：

|模型名称|输入图片大小|mAP<sup>pose<br>50-95|mAP<sup>pose<br>50|Speed<br><sup>CPU ONNX<br>(ms)|Speed<br><sup>A100 TensorRT<br>(ms)|params<br><sup>(M)|FLOPs<br><sup>(B)|
|:-|:-|:-|:-|:-|:-|:-|:-|
|YOLOv8n-pose|640|50.4|80.1|131.8|1.18|3.3|9.2|
|YOLOv8s-pose|640|60.0|86.2|233.2|1.42|11.6|30.2|
|YOLOv8m-pose|640|65.0|88.8|456.3|2.00|26.4|81.0|
|YOLOv8l-pose|640|67.6|90.0|784.5|2.59|44.4|168.6|
|YOLOv8x-pose|640|69.2|90.2|1607.1|3.73|69.4|263.2|
|YOLOv8x-pose-p6|1280|71.6|91.2|4088.7|10.04|99.1|1066.4|

### 1.3.4 旋转目标检测（Oriented Bounding Boxes Object Detection）

旋转目标检测进一步超越了物体检测，引入了额外的角度信息，以更准确地在图像中定位物体。

旋转目标检测器的输出是一组旋转的边界框，准确地包围图像中的物体，同时还包括每个框的类别标签和置信度分数。当我们需要在场景中识别感兴趣的物体，并且需要知道物体的精确位置和形状时，旋转目标检测是一个很好的选择。

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-04-12-11-57-05.png
    width=100%>
    <center></center>
</div></br>

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-04-12-11-57-14.png
    width=100%>
    <center></center>
</div></br>

> DOTA 数据集 v1.0 是一个专为航拍图像中目标检测任务设计的大规模数据集。这个数据集是目前最大的光学遥感图像数据集之一。DOTA 数据集 v1.0 共收录了 2806 张图像，每张图像的大小约为 4000×4000 像素，总共包含 188282 个目标实例。这些目标实例涵盖了各种不同的比例、方向和形状，使得数据集具有极高的真实性和挑战性。为了准确标注这些目标，数据集采用了旋转框的标记方式，即标注出每个目标的四个顶点，从而得到不规则四边形的边界框。这种标注方式相比传统的水平标注方法更为精确，能够减少大量的重叠区域。

|模型名称|size<br><sup>(pixels)|mAP<sup>test<br>50|Speed<br><sup>CPU ONNX<br>(ms)|Speed<br><sup>A100 TensorRT<br>(ms)|params<br><sup>(M)|FLOPs<br><sup>(B)|
|:-|:-|:-|:-|:-|:-|:-|
|YOLOv8n-obb|1024|78.0|204.77|3.57|3.1|23.3|
|YOLOv8s-obb|1024|79.5|424.88|4.07|11.4|76.3|
|YOLOv8m-obb|1024|80.5|763.48|7.61|26.4|208.6|
|YOLOv8l-obb|1024|80.7|1278.42|11.83|44.5|433.8|
|YOLOv8x-obb|1024|81.36|1759.10|13.23|69.5|676.7|

### 1.3.5 分类

图像分类是这三个任务中最简单的任务之一，它涉及将整个图像分类为预定义类别之一。

图像分类器的输出是一个单一的类别标签和置信度分数。图像分类在我们只需要知道图像属于哪个类别，而不需要知道该类别的物体位于何处或其确切形状时非常有用。

|模型名称|size<br><sup>(pixels)|acc<br><sup>top1|acc<br><sup>top5|Speed<br><sup>CPU ONNX<br>(ms)|Speed<br><sup>A100 TensorRT<br>(ms)|params<br><sup>(M)|FLOPs<br><sup>(B) at 640|
|:-|:-|:-|:-|:-|:-|:-|:-|
|YOLOv8n-cls|224|69.0|88.3|12.9|0.31|2.7|4.3|
|YOLOv8s-cls|224|73.8|91.7|23.4|0.35|6.4|13.5|
|YOLOv8m-cls|224|76.8|93.5|85.4|0.62|17.0|42.7|
|YOLOv8l-cls|224|76.8|93.5|163.0|0.87|37.5|99.7|
|YOLOv8x-cls|224|79.0|94.6|232.0|1.01|57.4|154.8|

# 2. Integrations

Ultralytics 团队与领先的人工智能平台进行了重要的集成，扩展了 Ultralytics 产品的功能，增强了数据集标注、训练、可视化和模型管理等任务。与 Roboflow、ClearML、Comet、Neural Magic 和 OpenVINO 进行了合作。通过了解这些工具，我们可以知道如何优化人工智能工作流程。

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-04-12-13-52-46.png
    width=100%>
    <center></center>
</div></br>

- **Roboflow**：使用 Roboflow 可以将自定义数据集标注并直接导出到 YOLOv8 进行训练。
- **ClearML**：使用 ClearML，可以自动跟踪、可视化甚至远程训练 YOLOv8，轻松管理和追踪模型的训练过程。
- **Comet**：Comet 是免费的，可以保存 YOLOv8 模型、恢复训练，并可交互地可视化和调试预测结果。
- **Neural Magic**：使用 Neural Magic DeepSparse，可以将 YOLOv8 推理速度提高多达 6 倍，加速模型推理过程。

# 3. Ultralytics HUB

通过 Ultralytics HUB，我们可以体验无缝的人工智能。它是一站式解决方案，包括数据可视化、YOLOv5 和 YOLOv8 🚀模型的训练和部署，无需编码。使用先进的平台和用户友好的 Ultralytics 应用程序，将图像转化为可操作的洞察力，并轻松实现人工智能愿景。

TODO。


## 1.4 YOLOv8 可以推理的格式

Ultralytics 团队的代码具有非常强大的功能，因此模型可以推理几乎所有的格式，如下所示：

|Source|Argument|Type|Notes|
|:-|:-|:-|:-|
|image|'image.jpg'|str or Path|单个图像文件|
|URL|'https://ultralytics.com/images/bus.jpg'|str|图像的URL|
|screenshot|'screen'|str|屏幕截图|
|PIL|Image.open('im.jpg')|PIL.Image|RGB通道的HWC格式|
|OpenCV|cv2.imread('im.jpg')|np.ndarray|BGR通道uint8（0-255）的HWC格式|
|numpy|np.zeros((640,1280,3))|np.ndarray|BGR通道uint8（0-255）的HWC格式|
|torch|torch.zeros(16,3,320,640)|torch.Tensor|RGB通道float32（0.0-1.0）的BCHW格式|
|CSV|'sources.csv'|str or Path|包含图像、视频或目录路径的CSV文件|
|video ✅|'video.mp4'|str or Path|MP4、AVI等格式的视频文件|
|directory ✅|'path/'|str or Path|包含图像或视频的目录路径|
|glob ✅|'path/*.jpg'|str|匹配多个文件的Glob模式使用*字符作为通配符|
|YouTube ✅|'https://youtu.be/LNwODJXcvt4'|str|指向YouTube视频的URL|
|stream ✅|'rtsp://example.com/media.mp4'|str|用于RTSP、RTMP、TCP或IP地址等流协议的URL|
|multi-stream ✅|'list.streams'|str or Path|*.streams文本文件，每行一个流URL，即8个流将以batch-size 8运行|

> `glob`是一种通配符模式，用于匹配指定规则的文件名。在 Linux 和 Unix 系统中，`glob` 也被用于匹配文件名。在 Python 中，`glob` 模块用于检索与指定模式匹配的文件/路径名。例如，`glob.glob('*.txt')` 将返回当前目录中所有以 `.txt` 结尾的文件名。

## 1.5 YOLOv8 推理结果的使用

### 1.5.1 获取推理结果 results

在 YOLOv8 中，模型的推理结果其实<font color='red'>是一个实例化类对象，所以它有自己的方法和属性</font>。

```python
from ultralytics import YOLO


# 加载模型
model = YOLO('pretrained_weights/yolov8n.pt')

# 让模型推理，我们可以得到结果
results = model(['img1.jpg', 'img2.jpg', 'img3.jpg'])
print(f"type(results): {type(results)}")  # <class 'list'>

# 接下来我们就可以处理结果了
for result in results:
    print(f"type(result): {type(result)}")  # <class 'ultralytics.engine.results.Results'>

    boxes = result.boxes  # [目标检测任务] bbox outputs
    masks = result.masks  # [分割任务] 分割得到的 masks
    keypoints = result.keypoints  # [关键点检测任务] 关键点
    probs = result.probs  # [分类任务] 类别概率
    
    print(f"type(boxes): {type(boxes)}")  # <class 'ultralytics.engine.results.Boxes'>
    print(f"type(masks): {type(masks)}")  # <class 'NoneType'>
    print(f"type(keypoints): {type(keypoints)}")  # <class 'NoneType'>
    print(f"type(probs): {type(probs)}")  # <class 'NoneType'>
```

我们可以看到，模型推理结果得到的 `results` 是一个 list，我们可以对其遍历得到 `result`。之后查看 `result` 的数据类型，是 `<class 'ultralytics.engine.results.Results'>` 的实例化对象，所以 `result` 会有一下方法和属性。再对 `result` 取 `.boxes`、`.masks`、`.keypoints`以及 `.probs` 可以取出不同任务的结果。在 ultralytics 项目中，默认的任务是目标检测，因此我们在查看数据类型的时候发现，只有 `boxes` 是 `<class 'ultralytics.engine.results.Boxes'>` 的实例化对象，其他的都是 `<class 'NoneType'>` 的实例化对象（即为 `None`）。

### 1.5.2 Results 实例化对象的属性和方法介绍

除了上述的属性外，`<class 'ultralytics.engine.results.Results'>` 的实例化对象 `result` 所有的属性如下所示：

|属性|Type|描述|
|:-|:-|:-|
|orig_img|`numpy.ndarray`|原始图像的 `numpy` 数组|
|orig_shape|`tuple`|以 `(高度，宽度)` 格式表示的原始图像形状|
|boxes|`Boxes, optional`|包含检测边界框的 `Boxes` 对象|
|masks|`Masks, optional`|包含检测掩模的 `Masks` 对象|
|probs|`Probs, optional`|包含分类任务每个类别的概率的 `Probs` 对象|
|keypoints|`Keypoints, optional`|包含每个对象检测到的关键点的 `Keypoints` 对象|
|speed|`dict`|每张图像的预处理、推理和后处理速度的字典，以毫秒为单位|
|names|`dict`|类别名称的字典|
|path|`str`|图像文件的路径|

因为 `result` 是一个类对象，所以它也有方法，所有方法如下：

|方法|返回值类型|描述|
|:-|:-|:-|
|`__getitem__()`|Results|返回指定索引的Results对象|
|`__len__()`|int|返回Results对象中检测结果的数量|
|`update()`|None|更新Results对象的boxes、masks和probs属性|
|`cpu()`|Results|返回所有张量都在CPU内存上的Results对象的副本|
|`numpy()`|Results|返回所有张量都作为numpy数组的Results对象的副本|
|`cuda()`|Results|返回所有张量都在GPU内存上的Results对象的副本|
|`to()`|Results|返回具有指定设备和dtype的张量的Results对象的副本|
|`new()`|Results|返回具有相同图像、路径和名称的新Results对象|
|`keys()`|List[str]|返回非空属性名称的列表|
|`plot()`|numpy.ndarray|绘制检测结果。返回带注释的图像的numpy数组|
|`verbose()`|str|返回每个任务的日志字符串|
|`save_txt()`|None|将预测保存到txt文件中|
|`save_crop()`|None|将裁剪的预测保存到save_dir/cls/file_name.jpg中|
|`tojson()`|None|将对象转换为JSON格式|

💡 **Tips**：对于得到的结果，我们也可以将它们转移到任意的设备中，详情如下：

```python
results = results.cuda()
results = results.cpu()
results = results.to('cpu')
results = results.numpy()
```

### 1.5.3 目标检测任务的 Boxes 实例化对象的属性和方法

接下来我们看看 `<class 'ultralytics.engine.results.Boxes'>` 的实例化对象 `boxes` 还有哪些操作。

以下是 Boxes 类的方法和属性的表格，包括它们的名称、类型和描述：

|名称|Type|描述|
|:-|:-|:-|
|`cpu()`|方法|将对象移动到CPU内存|
|`numpy()`|方法|将对象转换为numpy数组|
|`cuda()`|方法|将对象移动到CUDA内存|
|`to()`|方法|将对象移动到指定的设备|
|`xyxy`|属性 (torch.Tensor)|以xyxy格式返回边界框|
|`conf`|属性 (torch.Tensor)|返回边界框的置信度值|
|`cls`|属性 (torch.Tensor)|返回边界框的类别值|
|`id`|属性 (torch.Tensor)|返回边界框的跟踪ID（如果有）|
|`xywh`|属性 (torch.Tensor)|以xywh格式返回边界框|
|`xyxyn`|属性 (torch.Tensor)|以原始图像大小归一化的xyxy格式返回边界框|
|`xywhn`|属性 (torch.Tensor)|以原始图像大小归一化的xywh格式返回边界框|


💡 **Tips**：什么是 xyxy 格式、什么又是 xywh 格式？

在 YOLO 中，`xyxy` 格式和 `xywh` 格式都是用于表示物体边界框的两种常见格式。其中：
+ `xyxy` 格式指的是物体边界框的左上角和右下角的坐标，即 `(x1, y1, x2, y2)`；
+  `xywh` 格式则指的是物体边界框的中心点坐标、宽度和高度，即 `(x, y, w, h)`。

如果我们有一个边界框的 `xyxy` 坐标，我们可以使用以下公式将其转换为 `xywh` 格式：

$$
\begin{aligned}
    x &= \frac{x_1 + x_2}{2} \\
    y &= \frac{y_1 + y_2}{2} \\
    w &= x_2 - x_1 \\
    h &= y_2 - y_1
\end{aligned}
$$

反之，如果我们有一个边界框的 `xywh` 坐标，我们可以使用以下公式将其转换为 `xyxy` 格式：

$$
\begin{aligned}
    x_1 &= x - \frac{w}{2} \\
    y_1 &= y - \frac{h}{2} \\
    x_2 &= x + \frac{w}{2} \\
    y_2 &= y + \frac{h}{2}
\end{aligned}
$$



















# 知识来源

1. [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
2. [2023最新YOLOv8教程！计算机视觉从0到实战开发，推理及训练（代码实战）入门到精通！](https://www.bilibili.com/video/BV1zj411H7x9?vd_source=ac73c03faf1b37a5bc2296969f45cf7b)
3. 