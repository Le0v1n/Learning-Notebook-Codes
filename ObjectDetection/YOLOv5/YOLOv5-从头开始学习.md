# 1. 前置知识

## 1.1 YOLO 算法的基本思想

<div align=center>
    <img src=./imgs_markdown/2024-01-30-12-02-54.png
    width=50%>
</div>

首先通过特征提取网络对输入图像提取特征，得到一定大小的特征图，比如 13x13（相当于416x416 图片大小），然后将输入图像分成 13x13 个 grid cells：
- YOLOv3/v4：如果 GT 中某个目标的<font color='red'>中心</font>坐标落在哪个 grid cell 中，那么就由该 grid cell 来预测该目标。每个 grid cell 都会预测 3 个不同尺度的边界框。
- YOLOv5：不同于 YOLOv3/v4，其 GT <font color='blue'>可以跨层预测</font>，即有些 bbox（anchors）在多个预测层都算正样本，匹配数的正样本范围可以是 3-9 个。

预测得到的输出特征图有两个维度是提取到的特征的维度，比如 13x13，还有一个维度（深度）是 Bx(5+C)，其中：
 - B 表示每个 grid cell 预测的边界框的数量（YOLOv3/v4中是 3 个）
 - C 表示边界框的类别数（没有背景类，所以对于 VOC 数据集是 20）
 - 5 表示 4 个坐标信息和一个目标性得分（objectness score）

## 1.2 损失函数

1. **Classification Loss**：
    - 用于衡量模型对目标的分类准确性。
    - 计算方式通常使用交叉熵损失函数，该函数衡量模型的分类输出与实际类别之间的差异。
    - 对于 YOLOv5，每个目标都有一个对应的类别，分类损失量化了模型对每个目标类别的分类准确性。
2. **Localization Loss：定位损失（预测边界框与 GT 之间的误差）**
    - 用于衡量模型对目标位置的预测准确性。
    - YOLOv5 中采用的是均方差（Mean Squared Error，MSE）损失函数，衡量模型对目标边界框坐标的回归预测与实际边界框之间的差异。
    - 定位损失关注模型对目标位置的精确度，希望模型能够准确地定位目标的边界框。
3. **Confidence Loss：置信度损失（框的目标性 <=> Objectness of the box）**
    - 用于衡量模型对目标存在与否的预测准确性。
    - YOLOv5 中采用的是二元交叉熵损失函数，该函数衡量模型对目标存在概率的预测与实际目标存在的二元标签之间的差异。
    - 置信度损失考虑了模型对每个边界框的目标置信度以及是否包含目标的预测。该损失鼓励模型提高对包含目标的边界框的预测概率，同时减小对不包含目标的边界框的预测概率。

总的损失函数：

$$
\rm Loss = \alpha \times Classification Loss + \beta \times Localization Loss + \gamma \times Confidence Loss
$$

## 1.3 PyTorch2ONNX

Netron 对 `.pt` 格式的兼容性不好，直接打卡无法显示整个网络。因此我们可以使用 YOLOv5 中的 `models/export.py` 脚本将 `.pt` 权重转换为 `.onnx` 格式，再使用 Netron 打开就可以完整地查看 YOLOv5 的整体架构了。

```bash
python export.py \
    --weights weights/yolov5s.pt \
    --imgsz 640 \
    --batch-size 1 \
    --device cpu \
    --simplify \
    --include onnx
```

> 💡 详细可选参数见 `export.py` 文件

## 1.4 YOLOv5 模型结构图

<div align=center>
    <img src=/ObjectDetection/YOLOv5/yolov5-arch.png
    width=100%>
</div>

> 图片来源：霹雳吧啦Wz

# 2. 配置文件

在 `models` 中的 `.yaml` 文件是模型的配置文件

```
models
├── __init__.py
├── tf.py
├── yolo.py
├── yolov5l.yaml
├── yolov5m.yaml
├── yolov5n.yaml
├── yolov5s.yaml
└── yolov5x.yaml
```

我们以 `yolov5s.yaml` 为例展开讲解。

## 2.1 模型深度系数 depth_multiple 和宽度系数 width_multiple

```yaml
# Parameters
nc: 80 # number of classes | 类别数
depth_multiple: 0.33 # model depth multiple | 模型深度: 控制 BottleneckCSP 数
width_multiple: 0.50 # layer channel multiple | 模型宽度: 控制 Conv 通道个数（卷积核数量）
```

- `depth_multiple` 表示 BottleneckCSP、C3 等层缩放因子，将所有的 BottleneckCSP、C3等 模块的 Bottleneck 子模块 乘上该参数得到最终的 Bottleneck 子模块个数
- `width_multiple` 表示卷积通道的缩放因子，就是将配置里的 `backbone` 和 `head` 部分（<font color='red'>其实就是所有的</font>）有关 `Conv` 的通道都需要乘上该系数

通过 `depth_multiple` 和 `width_multiple` 参数可以实现不同复杂度的模型设计：yolov5x、yolov5s、yolov5n、yolov5m、yolov5l。

<details><summary>BottleneckCSP 和 C3 的结构示意图</summary>

<div align=center>
    <img src=./imgs_markdown/2024-01-30-15-06-54.png
    width=80%>
    <center>BotteleneckCSP 结构</center>
</div>

> BotteleneckCSP 图片来源: [深入浅出Yolo系列之Yolov5核心基础知识完整讲解](https://zhuanlan.zhihu.com/p/172121380?utm_oi=551376804724793344)

<div align=center>
    <img src=./imgs_markdown/2024-01-30-13-48-51.png
    width=50%>
</div>

<div align=center>
    <img src=./imgs_markdown/2024-01-30-13-49-36.png
    width=50%>
    <center>C3 结构</center>
</div>

</details>

## 2.2 anchors | 先验框大小

```yaml
anchors:
  - [10, 13, 16, 30, 33, 23] # P3/8
  - [30, 61, 62, 45, 59, 119] # P4/16
  - [116, 90, 156, 198, 373, 326] # P5/32
```

上面就定义了三种尺寸的先验框的大小，其中：

- `P3/8`：`P3` 是层的名称，`8` 表示此时特征图经过的下采样大小 → P3 特征图此时已经过了 8 倍下采样
- `P4/16`：P4 特征图此时已经过了 16 倍下采样
- `P5/32`：P5 特征图此时已经过了 32 倍下采样

> 在 YOLOv5 中，P3 代表 Feature Pyramid Network (FPN) 的第三个级别。FPN 是一种用于目标检测的特征提取网络结构，它通过在不同层级的特征图上应用卷积和上采样操作，以获取具有不同尺度和语义信息的特征图。这些特征图可以用于检测不同大小的目标。
> 
> 在这个模型配置文件中，P3/8 表示 P3 层在输入图像上的缩放因子为 8。缩放因子指的是在输入图像上的每个像素点在 P3 层特征图上所对应的尺寸。通过这种缩放，可以使得 P3 层特征图的尺寸相对于输入图像缩小 8 倍。这种缩放操作帮助模型捕获不同尺度的目标信息。

## 2.3 backbone

```yaml
# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [
    [-1, 1, Conv, [64, 6, 2, 2]], # 0-P1/2
    [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
    [-1, 3, C3, [128]],
    [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
    [-1, 6, C3, [256]],
    [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
    [-1, 9, C3, [512]],
    [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
    [-1, 3, C3, [1024]],
    [-1, 1, SPPF, [1024, 5]], # 9
  ]
```

首先第一行的备注信息已经告诉我们了，这个 backbone 是 YOLOv5 和 YOLOv6 的 backbone。第二行中有对每一列的说明，其中：
- `from`：表示输入的来源。-1 表示前一层的输出作为输入。
- `number`：表示重复使用该模块的次数。
- `module`：表示使用的特征提取模块类型。
- `args`：表示模块的参数：
  - Conv 层：输出通道数、卷积核大小、步幅和填充
  - C3 层：输出通道数
  - SPPF 层：表示输出通道数和池化的 `kernel_size`。

> 💡 注意：
> - 在之前的版本（v4.0）中，backbone 的第一层是一个 Focus 层，但现在是一个卷积层。
> - 对于 C3 层而言，如果重复了 3 次，且 `stride=2`，那么<font color='red'>只有第一个 C3 模块会进行两倍下采样</font>，<font color='green'>剩下的两个 C3 模块不会进行下采样操作</font>

---

<font color='blue'><b>〔与模型深度系数 depth_multiple 和宽度系数 width_multiple 的联系〕</b></font>

前面说过了 `depth_multiple` 和 `width_multiple` 这两个参数的作用，对于 YOLOv5-s 的 `C3` 层而言，此时的 `depth_multiple=0.33`，那么第二列的 `C3` 层个数并不是实际上的数量，实际上的数量还得乘上 `depth_multiple`：

```yaml
# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [
    [-1, 1, Conv, [64, 6, 2, 2]], # 0-P1/2
    [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
    [-1, 3, C3, [128]],  # 3*0.33=0.99 ---------> 实际使用1个C3
    [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
    [-1, 6, C3, [256]],  # 6*0.33=1.98 ---------> 实际使用2个C3
    [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
    [-1, 9, C3, [512]],  # 9*0.33=2.97 ---------> 实际使用3个C3
    [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
    [-1, 3, C3, [1024]],  # 3*0.33=0.99 ---------> 实际使用1个C3
    [-1, 1, SPPF, [1024, 5]], # 9
  ]
```

<kbd><b>Q</b>uestion</kbd>：这个计算是怎么进行的？
<kbd><b>A</b>nswer</kbd>：在 `models/yolo.py` 的 `parse_model()` 函数中有写：

```python
# 对 backbone 和 head 中的所有层进行遍历
for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  
    # f <-> from：表示输入的来源。-1 表示前一层的输出作为输入。
    # n <-> number：表示重复使用该模块的次数。
    # m <-> module：表示使用的特征提取模块类型。
    # args：表示模块的参数：

    # 将字符串转换为对应的代码名称（不懂的看一下 eval 函数）
    m = eval(m) if isinstance(m, str) else m  

    # 遍历每一层的参数args
    for j, a in enumerate(args):
        # j: 参数的索引
        # a: 具体的参数
        with contextlib.suppress(NameError):
            # 将数字或字符长转换为代码
            args[j] = eval(a) if isinstance(a, str) else a  # eval strings

    # 先将所有的 number 乘上 深度系数
    n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
```

这个根据向上取整的操作，并确保结果至少为 1

那么对于 `width_multiple` 系数而言，也是一样的（在 YOLOv5s 中, `width_multiple=0.50`）：

```yaml
# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [
    [-1, 1, Conv, [64, 6, 2, 2]], # 0-P1/2  ----------> 64  * 0.5 = 32
    [-1, 1, Conv, [128, 3, 2]], # 1-P2/4    ----------> 128 * 0.5 = 64
    [-1, 3, C3, [128]],  #                  ----------> 128 * 0.5 = 64
    [-1, 1, Conv, [256, 3, 2]], # 3-P3/8    ----------> 256 * 0.5 = 128
    [-1, 6, C3, [256]],  #                  ----------> 256 * 0.5 = 128
    [-1, 1, Conv, [512, 3, 2]], # 5-P4/16   ----------> 512 * 0.5 = 256
    [-1, 9, C3, [512]],  #                  ----------> 512 * 0.5 = 256
    [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32  ----------> 1024* 0.5 = 512
    [-1, 3, C3, [1024]],  #                 ----------> 1024* 0.5 = 512
    [-1, 1, SPPF, [1024, 5]], # 9           ----------> 1024* 0.5 = 512
  ]
```

意思就是说，将所有的卷积层都乘上 `width_multiple`，那我们看一下代码细节（还是在 `models/yolo.py -> parse_model()` 中）：

```python
# 对 backbone 和 head 中的所有层进行遍历
for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  
    # f <-> from：表示输入的来源。-1 表示前一层的输出作为输入。
    # n <-> number：表示重复使用该模块的次数。
    # m <-> module：表示使用的特征提取模块类型。
    # args：表示模块的参数：

    # 将字符串转换为对应的代码名称（不懂的看一下 eval 函数）
    m = eval(m) if isinstance(m, str) else m  

    # 遍历每一层的参数args
    for j, a in enumerate(args):
        # j: 参数的索引
        # a: 具体的参数
        with contextlib.suppress(NameError):
            # 将数字或字符长转换为代码
            args[j] = eval(a) if isinstance(a, str) else a  # eval strings

    # 先将所有的 number 乘上 深度系数
    n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain

    # 判断当前模块是否在这个字典中
    if m in {
        Conv,  # Conv + BN + SiLU
        GhostConv,  # 华为在 GhostNet 中提出的Ghost卷积
        Bottleneck,  # ResNet同款
        GhostBottleneck,  # 将其中的3x3卷积替换为GhostConv
        SPP,  # Spatial Pyramid Pooling
        SPPF,  # SPP + Conv
        DWConv,  # 深度卷积
        MixConv2d,  # 一种多尺度卷积层，可以在不同尺度上进行卷积操作。它使用多个不同大小的卷积核对输入特征图进行卷积，并将结果进行融合
        Focus,  # 一种特征聚焦层，用于减少计算量并增加感受野。它通过将输入特征图进行通道重排和降采样操作，以获取更稠密和更大感受野的特征图
        CrossConv,  # 一种交叉卷积层，用于增加特征图的多样性。它使用不同大小的卷积核对输入特征图进行卷积，并将结果进行融合
        BottleneckCSP,  # 一种基于残差结构的卷积块，由连续的Bottleneck模块和CSP（Cross Stage Partial）结构组成，用于构建深层网络，提高特征提取能力
        C3,  # 一种卷积块，由三个连续的卷积层组成。它用于提取特征，并增加网络的非线性能力
        C3TR,  # C3TR是C3的变体，它在C3的基础上添加了Transpose卷积操作。Transpose卷积用于将特征图的尺寸进行上采样
        C3SPP,  # C3SPP是C3的变体，它在C3的基础上添加了SPP操作。这样可以在不同尺度上对特征图进行池化，并增加网络的感受野
        C3Ghost,  # C3Ghost是一种基于C3模块的变体，它使用GhostConv代替传统的卷积操作
        nn.ConvTranspose2d,  # 转置卷积
        DWConvTranspose2d,  # DWConvTranspose2d是深度可分离的转置卷积层，用于进行上采样操作。它使用逐点卷积进行特征图的通道之间的信息整合，以减少计算量
        C3x,  # C3x是一种改进的C3模块，它在C3的基础上添加了额外的操作，如注意力机制或其他模块。这样可以进一步提高网络的性能
    }:
        c1, c2 = ch[f], args[0]  # c1: 卷积的输入通道数, c2: 卷积的输出通道数 | ch[f] 上一次的输出通道数（即本层的输入通道数），args[0]：配置文件中想要的输出通道数
        if c2 != no:  # if not output
            c2 = make_divisible(c2 * gw, ch_mul)  # 让输出通道数*width_multiple

        args = [c1, c2, *args[1:]]  # 此时的c2已经是修改后的c2乘上width_multiple的c2了 | *args[1:]将其他非输出通道数的参数解包

        # 如果当前层是 BottleneckCSP, C3, C3TR, C3Ghost, C3x 中的一种（这些结构都有 Bottleneck 结构）
        if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
            args.insert(2, n)  # number of repeats | 需要让Bottleneck重复n次
            n = 1  # 重置n（其他层没有 Bottleneck 的模块不需要重复）
    # 如果是BN层
    elif m is nn.BatchNorm2d:
        args = [ch[f]]  # 确定输出通道数
    
    # 如果是 Concat 层
    elif m is Concat:
        c2 = sum(ch[x] for x in f)  # Concat是按着通道维度进行的，所以通道会增加
```

## 2.4 head

```yaml
# YOLOv5 v6.0 head
head: [
    [-1, 1, Conv, [512, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [[-1, 6], 1, Concat, [1]], # cat backbone P4
    [-1, 3, C3, [512, False]], # 13

    [-1, 1, Conv, [256, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [[-1, 4], 1, Concat, [1]], # cat backbone P3
    [-1, 3, C3, [256, False]], # 17 (P3/8-small)

    [-1, 1, Conv, [256, 3, 2]],
    [[-1, 14], 1, Concat, [1]], # cat head P4
    [-1, 3, C3, [512, False]], # 20 (P4/16-medium)

    [-1, 1, Conv, [512, 3, 2]],
    [[-1, 10], 1, Concat, [1]], # cat head P5
    [-1, 3, C3, [1024, False]], # 23 (P5/32-large)

    [[17, 20, 23], 1, Detect, [nc, anchors]], # Detect(P3, P4, P5)
  ]
```

💡 **Tips**：

- 列的定义和 backbone 是一样的
- 不像 YOLOv3 那样，作者区分了 Neck 和 Head。YOLOv5 的作者没有做出区分，只有 Head，所以在 Head 部分中包含了 PANet 和 Detect 部分。

<kbd><b>Q</b>uestion</kbd>：`Concat` 怎么理解？
<kbd><b>A</b>nswer</kbd>：我们看下面的图。

<div align=center>
    <img src=./imgs_markdown/2024-01-30-16-48-02.png
    width=100%>
</div>

这里的 Concat 就是把浅层的特征图与当前特征图进行拼接（沿通道维度），我们看一下源码（在 `models/common.py -> Concat` 中）：

```python
class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        # 这里的 x 是一个 list，所以可以有多个 Tensor 进行拼接
        return torch.cat(x, self.d)
```

这里需要注意的其实就是 `from`，即谁和谁拼接？下面是解释：

```yaml
# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [
    [-1, 1, Conv, [64, 6, 2, 2]],                # 0-P1/2
    [-1, 1, Conv, [128, 3, 2]],                  # 1-P2/4
    [-1, 3, C3, [128]],                          # 2
    [-1, 1, Conv, [256, 3, 2]],                  # 3-P3/8
    [-1, 6, C3, [256]],                          # 4
    [-1, 1, Conv, [512, 3, 2]],                  # 5-P4/16
    [-1, 9, C3, [512]],                          # 6
    [-1, 1, Conv, [1024, 3, 2]],                 # 7-P5/32
    [-1, 3, C3, [1024]],                         # 8
    [-1, 1, SPPF, [1024, 5]],                    # 9
  ]

# YOLOv5 v6.0 head
head: [
    [-1, 1, Conv, [512, 1, 1]],                   # 10
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],   # 11
    [[-1, 6], 1, Concat, [1]], # cat backbone P4  # 12
    [-1, 3, C3, [512, False]],                    # 13

    [-1, 1, Conv, [256, 1, 1]],                   # 14
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],   # 15
    [[-1, 4], 1, Concat, [1]], # cat backbone P3  # 16
    [-1, 3, C3, [256, False]],                    # 17 (P3/8-small)

    [-1, 1, Conv, [256, 3, 2]],                   # 18
    [[-1, 14], 1, Concat, [1]], # cat head P4     # 19
    [-1, 3, C3, [512, False]],                    # 20 (P4/16-medium)

    [-1, 1, Conv, [512, 3, 2]],                   # 21
    [[-1, 10], 1, Concat, [1]], # cat head P5     # 22
    [-1, 3, C3, [1024, False]],                   # 23 (P5/32-large)

    [[17, 20, 23], 1, Detect, [nc, anchors]], # Detect(P3, P4, P5)  # 24
  ]
```

我们看第一个 Concat：`[[-1, 6], 1, Concat, [1]]`：
- `-1` 表示上一层（即 Concat 的前一层）
- `6` 表示第 6 层，即 backbone 中的 `[-1, 9, C3, [512]]`。

剩下的以此类推。

> ⚠️ <font color='red'>这里的索引是从 0 开始的</font>

在 Head 中，`P` 其实对应的是检测头对应的输出层。比如说 `P3` 就是 8 倍下采样的输出层。我们常用的是 `P3+P4+P5`。为了捕获更小的目标，我们可以使用 `models/hub/yolov5-p2.yaml` 这个模型：

```yaml
# YOLOv5 v6.0 head with (P2, P3, P4, P5) outputs
head: [
    [-1, 1, Conv, [512, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [[-1, 6], 1, Concat, [1]], # cat backbone P4
    [-1, 3, C3, [512, False]], # 13

    [-1, 1, Conv, [256, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [[-1, 4], 1, Concat, [1]], # cat backbone P3
    [-1, 3, C3, [256, False]], # 17 (P3/8-small)

    [-1, 1, Conv, [128, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [[-1, 2], 1, Concat, [1]], # cat backbone P2
    [-1, 1, C3, [128, False]], # 21 (P2/4-xsmall)

    [-1, 1, Conv, [128, 3, 2]],
    [[-1, 18], 1, Concat, [1]], # cat head P3
    [-1, 3, C3, [256, False]], # 24 (P3/8-small)

    [-1, 1, Conv, [256, 3, 2]],
    [[-1, 14], 1, Concat, [1]], # cat head P4
    [-1, 3, C3, [512, False]], # 27 (P4/16-medium)

    [-1, 1, Conv, [512, 3, 2]],
    [[-1, 10], 1, Concat, [1]], # cat head P5
    [-1, 3, C3, [1024, False]], # 30 (P5/32-large)

    [[21, 24, 27, 30], 1, Detect, [nc, anchors]], # Detect(P2, P3, P4, P5)
  ]
```

## 2.5 不同规格模型配置

