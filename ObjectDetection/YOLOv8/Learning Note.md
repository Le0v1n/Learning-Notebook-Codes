# 1. 概述

## 1.1 项目介绍

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 是一种前沿、最先进 (SOTA) 的模型，它在之前的 YOLO 版本的成功基础上引入了新功能和改进，以提高性能和灵活性。YOLOv8 旨在快速、准确、易于使用，是广泛应用于目标检测和跟踪、实例分割、图像分类和姿态估计等任务的优秀选择。

## 1.2 快速上手

与之前的 YOLOv5 不同的是，YOLOv8 是在一个名为 Ultralytics 项目下，该项目将该团队之前制作的 YOLOv3、YOLOv5整合到了一起，并添加了 YOLOv8，所以我们首先需要安装一个非常独特的库 —— ultralytics：

```bash
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
```

在我们安装好 ultralytics 库并下载 ultralytics 项目后，可以直接使用命令行（Command Line Interface, CLI）进行快速推理一张图片、视频、视频流、摄像头等等：

```bash
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

除了使用命令行来进行推理外，我们也可以写一个 Python 脚本来进行相同的操作：

```python
from ultralytics import YOLO


# ============= 加载模型 =============
model = YOLO('yolov8n.yaml')  # 通过yaml文件新建一个模型(根据yaml文件中的模型定义自动搭建一个模型)
model = YOLO('yolov8n.pt')  # 加载一个训练好的模型（直接从pt文件中读取模型架构从而搭建模型）

# ============= 模型训练 =============
model.train(data='coco128.yaml', epochs=3)  # 训练 coco128.yaml 中定义的数据集，并且 epochs 为 3
metrics = model.val()  # 让模型在验证集中测试模型性能
 
# ============= 模型训练完毕后做的事情 =============
results = model("https://ultralytics.com/images/bus.jpg")  # 让模型推理一张 web 图片
path = model.export(format='onnx')  # 模型转换为 onnx 格式（不影响原来的 .pt 文件）
```

## 1.3 模型

YOLOv8 Detect、Segment 和 Pose 模型是在 COCO 数据集上预训练的，官网提供了这些模型，以及在 ImageNet 数据集上预训练的 YOLOv8 Classify 模型。所有 Detect、Segment 和 Pose 模型都提供跟踪模式。

我们先看一下 YOLOv8 的检测模型：

|模型名称|输入图片大小|mAP@50-95|CPU@ONNX Speed (ms)| A100@TensorRT (ms)|params (M)|FLOPs (B)|
|:-|:-|:-|:-|:-|:-|:-|
|YOLOv8n|640|37.3|80.4|0.99|3.2|8.7|
|YOLOv8s|640|44.9|128.4|1.20|11.2|28.6|
|YOLOv8m|640|50.2|234.7|1.83|25.9|78.9|
|YOLOv8l|640|52.9|375.2|2.39|43.7|165.2|
|YOLOv8x|640|53.9|479.1|3.53|68.2|257.8|

其中，n、s、m、l、x的意思如下：

+ `YOLOv8n --> YOLOv8 Nano`：非常小
+ `YOLOv8s --> YOLOv8 Small`：小
+ `YOLOv8m --> YOLOv8 Medium`：中
+ `YOLOv8l --> YOLOv8 Large`：大
+ `YOLOv8x --> YOLOv8 Extra Large`：非常大


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