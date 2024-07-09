<center><font size=12>Deep High-Resolution Representation Learning for Human Pose Estimation</font></center>

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-07-07-22-20-00.png
    width=100%></br><center></center>
</div>

- **论文地址**：[Deep High-Resolution Representation Learning for Human Pose Estimation](https://arxiv.org/abs/1902.09212)
- **源码地址**：[deep-high-resolution-net.pytorch](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)

💡 **注意**：该网络主要是针对<font color='red'><b>单一个体</b></font>的姿态评估（即输入网络的图像中应该只有一个人体目标）。

> 图中展示的多人关键点检测的实现是：先经过一个行人检测模型，之后将每个行人抠出来，再送入HRNet，之后HRNet将这个人的关键点检测出来。最后将所有检测出来的关键点再还原会原图。

# 0. 摘要

这是Deep High-Resolution Representation Learning for Human Pose Estimation的官方PyTorch实现。在这项工作中，我们对学习可靠的高分辨率表示的人体姿态估计问题感兴趣。大多数现有方法从高到低分辨率网络产生的低分辨率表示中恢复高分辨率表示。相反，我们提出的网络在整个过程中保持高分辨率表示。我们从高分辨率子网络作为第一阶段开始，逐步添加高到低分辨率子网络形成更多阶段，并并行连接多分辨率子网络。我们进行重复的多尺度融合，使得每个高到低分辨率表示反复从其他并行表示接收信息，从而产生丰富的高分辨率表示。结果，预测的关键点热图可能更准确，空间上更精确。我们通过在两个基准数据集上的优越姿态估计结果来证明我们网络的有效性：COCO关键点检测数据集和MPII人体姿态数据集。

# 1. 前言

对于Human Pose Estimation任务，现在基于深度学习的方法主要有两种：

- 基于回归（regressing）的方式，即直接预测每个关键点的位置坐标。
- 基于热力图（heatmap）的方式，即针对每个关键点预测一张热力图（预测出现在每个位置上的分数）。

当前检测效果最好的一些方法基本都是基于heatmap的，所以HRNet也是采用基于heatmap的方式。

# 2. HRNet网络结构

下图是[霹雳吧啦Wz](https://space.bilibili.com/18161609)根据阅读项目源码绘制的关于HRNet-W32的模型结构简图，在论文中除了提出HRNet-W32外还有一个HRNet-W48的版本，两者区别仅仅在每个模块所采用的通道个数不同，网络的整体结构都是一样的。而该论文的核心思想就是不断地去融合不同尺度上的信息，也就是论文中所说的Exchange Blocks。

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-07-07-22-40-23.png
    width=100%></br><center>HRNet-W32网络结构图</center>
</div>

通过上图可以看出，HRNet首先通过两个卷积核大小为`3x3`步距为`2`的卷积层（后面都跟有BN以及ReLU）共下采样了4倍。然后通过`Layer1`模块，这里的`Layer1`其实和之前讲的ResNet中的`Layer1`类似，就是重复堆叠`Bottleneck`，注意这里的`Layer1`只会调整通道个数，并不会改变特征层大小。下面是实现`Layer1`时所使用的代码。

```python
# Stage1
downsample = nn.Sequential(
    nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
)
self.layer1 = nn.Sequential(
    Bottleneck(64, 64, downsample=downsample),
    Bottleneck(256, 64),
    Bottleneck(256, 64),
    Bottleneck(256, 64)
)
```

通过霹雳吧啦Wz绘制的HRNet结构图我们可以发现，HRNet其实和我们之前讲过的语义分割模型很像，使用各种上采样和下采样，目的是让特征图实现特征融合（$\oplus$）。最后在`Stage4`阶段将所有的特征图都融合在一起，再经过`BN`和`Conv2d`得到最终的输出结果。

最后的输出为64×48×17，输入图像大小为256×192×3，经过一系列的特征提取，最后经过了4倍下采样得到最终的输出。为什么最后要有17呢？这是因为COCO数据集的人体姿态估计标签有17个类别，因此是17。下面是COCO数据集不同标签的含义：

| Index | label          | Chinese | Index | label          | Chinese |
| :---: | :------------- | :------ | :---: | :------------- | :------ |
|   0   | nose           | 鼻子    |   9   | left_wrist     | 左腕    |
|   1   | left_eye       | 左眼    |  10   | right_wrist    | 右腕    |
|   2   | right_eye      | 右眼    |  11   | left_hip       | 左髋    |
|   3   | left_ear       | 左耳    |  12   | right_hip      | 右髋    |
|   4   | right_ear      | 右耳    |  13   | left_knee      | 左膝    |
|   5   | left_shoulder  | 左肩    |  14   | right_knee     | 右膝    |
|   6   | right_shoulder | 右肩    |  15   | left_ankle     | 左踝    |
|   7   | left_elbow     | 左肘    |  16   | right_ankle    | 右踝    |
|   8   | right_elbow    | 右肘    |       |                |         |

> 💡 由于HRNet使用了很多ResNet的组件，因此这个网络必定不会很轻量型。

# 3. 预测结果（heatmap）的可视化

关于预测得到的heatmap（热力图）听起来挺抽象的，为了方便大家理解，[霹雳吧啦Wz](https://space.bilibili.com/18161609)画了下面这幅图。

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-07-07-22-59-32.png
    width=100%></br><center></center>
</div>

首先，左边是输入网络的预测图片，大小为256×192，为了保证原图像比例，在两侧进行了padding。右侧是从预测结果heatmap（64×48×17）中提取出的部分关键点对应的预测信息（48×17×1）。上面有提到过，网络最终输出的heatmap分辨率是原图经过4倍下采样大小，所以高宽分别对应的是64和48。接着对每个关键点对应的预测信息求最大值的位置，即预测`score`最大的位置，作为预测关键点的位置，映射回原图就能得到原图上关键点的坐标（上图有画出每个预测关键点对应原图的位置）。

但在原论文中，对于每个关键点并不是直接取`score`最大的位置（如果为了方便直接取其实也没太大影响）。在原论文的4.1章节中有提到：

> Each keypoint location is predicted by adjusting the highest heatvalue location with a quarter offset in the direction from the highest response to the second highest response.
>
> 每个关键点的位置是通过调整最高响应位置，以从最高响应到第二高响应方向的四分之一偏移量来预测的。

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-07-09-14-18-17.png
    width=50%></br><center></center>
</div>

假设对于某一关键点的预测heatmap如上所示，根据寻找最大`score`可以找到坐标(3, 3)点，接着分别对比该点左右两侧（x方向），上下两侧（y方向）的`score`。比如说先看左右两侧，明显右侧的score比左侧的大（蓝色越深代表score越大），所以最终预测的x坐标向右侧偏移0.25故最终x=3.25，同理上侧的score比下侧大，所以y坐标向上偏移0.25故最终y=2.75。

最后把每个关键点绘制在原图上，就得到如下图所示的结果。

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-07-09-14-19-47.png
    width=20%></br><center></center>
</div>

# 4. Loss的计算

HRNet训练采用的损失就是均方误差Mean Squared Error（MSE），MSE是衡量估计或预测值与真实值之间差异的一种常用方法。MSE的公式如下：

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2
$$

其中：
- $n$是样本数量。
- $Y_i$是第$i$个真实值。
- $\hat{Y}_i$是第$i$个预测值。
- $\sum$表示求和。

MSE的值越小，表示预测值与真实值之间的差异越小，预测或估计的准确度越高。

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：通过前面讲的内容我们知道网络预测的最终结果是针对每个关键点的heatmap，<font color='red'><b>那训练时对应的GT又是什么呢？</b></font>
🥳 𝑨𝒏𝒔𝒘𝒆𝒓：根据标注信息我们是可以得知每个关键点的坐标的（原图尺度），接着将坐标都除以4（缩放到heatmap尺度）再进行四舍五入。针对每个关键点，我们先生成一张值全为0的heatmap，然后将对应关键点坐标处填充1就得到下面左侧的图片。

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-07-09-14-24-57.png
    width=50%></br><center></center>
</div>

如果直接拿左侧的heatmap作为GT去训练网络的话，你会发现网络很难收敛，这是因为每个关键点只有一个点为正样本，其他64×48-1=3071个点都是负样本，<font color='red'><b>正负样本极度不均匀</b></font>。

为了解决这个问题一般会以关键点坐标为中心应用一个2D的高斯分布（没有做标准化处理）得到如右图所示的GT。利用这个GT heatmap配合网络预测的heatmap就能计算MSE损失了。

下面这幅图是某张真实训练样本（左侧）对应nose关键点的GT heatmap（右侧）。

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-07-09-14-27-30.png
    width=50%></br><center></center>
</div>

我们知道如何计算每个关键点对应的损失后还需要留意一个小细节。代码中在计算总损失时，并不是直接把每个关键点的损失进行相加，而是在相加前对于每个点的损失分别乘上不同的权重。下面给出了每个关键点的名称以及所对应的权重。

```python
"kps": ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder",
        "left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle"]
"kps_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                1.2, 1.2, 1.5, 1.5, 1.0, 1.0, 
                1.2, 1.2, 1.5, 1.5]
```

| 索引  | 关键点名称     | 权重 | 索引  | 关键点名称  | 权重 |
| :---: | :------------- | :--- | :---: | :---------- | :--- |
|   0   | nose           | 1.0  |   9   | left_wrist  | 1.5  |
|   1   | left_eye       | 1.0  |  10   | right_wrist | 1.5  |
|   2   | right_eye      | 1.0  |  11   | left_hip    | 1.0  |
|   3   | left_ear       | 1.0  |  12   | right_hip   | 1.0  |
|   4   | right_ear      | 1.0  |  13   | left_knee   | 1.2  |
|   5   | left_shoulder  | 1.0  |  14   | right_knee  | 1.2  |
|   6   | right_shoulder | 1.0  |  15   | left_ankle  | 1.5  |
|   7   | left_elbow     | 1.2  |  16   | right_ankle | 1.5  |
|   8   | right_elbow    | 1.2  |       |             |      |

# 5. 评价准则

在目标检测（Object Detection）任务中可以通过IoU（Intersection over Union）作为预测bbox和GT bbox之间的重合程度或相似程度。在关键点检测（Keypoint Detection）任务中一般用OKS（Object Keypoint Similarity）来表示预测keypoints与GT keypoints的相似程度，其值域在$[0, 1]$之间，越靠近1表示相似度越高。

$$
\mathrm{OKS} = \frac{\sum_i [e^{\frac{-d_i^2}{2s^2k_i^2}}\delta(v_i > 0)]}{\sum_i [\delta(v_i > 0)]}
$$

其中：

- $i$代表第$i$个关键点。
- $v_i$代表第$i$个关键点的可见性，这里的$v_i$是由GT提供的。
  - $v_i=0$表示该点一般是在图像外无法标注。
  - $v_i=1$表示虽然该点不可见但大概能猜测出位置（比如人侧着站时虽然有一只耳朵被挡住了，但大概也能猜出位置）。
  - $v_i=2$表示该点可见的（那么就意味着这个点应该被标注出来，也应该被模型预测出来）。
- $\delta(x)$表示当$x$为`True`时值为1，$x$为`False`时值为0。通过上面公式可知，OKS只计算GT中标注出的点，即$v_i>0$的所有关键点。
- $d_i$为第$i$个预测关键点与对应GT之间的欧氏距离。
- $s$为目标面积的平方根，原话：*scale s which we define as the square root of the object segment area*，这里的面积指的是分割面积。该数据在COCO数据集标注信息中都是有提供的。
- $k_i$是用来控制关键点类别$i$的衰减常数，原话：*κi is a per-keypont constant that controls falloff*，这个常数是在验证集（5000张）上统计得到的。

> 在MS COCO官网中有详细介绍OKS指标，详情参考: [https://cocodataset.org/#keypoints-eval](https://cocodataset.org/#keypoints-eval)，也可参考本文的[前置知识](#前置知识)。

# 6. 其他

## 6.1 代码

如果想要学习HRNet代码的话，不太建议直接去读官方源码。因为环境配置有些小问题，而且看起来令人头大。建议看霹雳吧啦Wz提供的HRNet仓库代码，他对原仓库代码做了一些修改，并加了很多注释，学习起来会更方便点，代码链接：[https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_keypoint/HRNet](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_keypoint/HRNet)

## 6.2 数据增强

在论文中作者采用的数据增强有：随机旋转（在 $[-45°,45°]$ 之间），随机缩放（在 $[0.65, 1.35]$ 之间），随机水平翻转以及half body（有一定概率会对目标进行裁剪，只保留半身关键点，上半身或者下半身）。在源码中，作者主要是通过仿射变换来实现的以上操作，如果对仿射变换不太了解看代码会比较吃力。

> 推荐博文：[数据增强中的仿射变换：旋转，缩放，平移以及错切(shear)](https://blog.csdn.net/weixin_44878336/article/details/124902173)。

## 6.3 注意输入图片比例

假设对于输入网络图片固定尺寸是256×192（height:width = 4:3），但要预测的人体目标的高宽比不是4:3，<font color='red'><b>此时千万不要直接简单粗暴的拉伸到256×192（如下图左侧所示）</b></font>。正确的方法有两种：

1. 保持目标原比例缩放到对应尺度然后再进行相应的padding（如下图中间所示，由于目标的height:width > 4:3，所以保持原比例将height缩放到256，然后在图片width两测进行padding得到256×192）。
2. 如果拥有原始图像的上下文信息的话可以直接在原图中固定height（目标height:width > 4:3的情况）然后调整width保证height:width = 4:3，再重新裁剪目标并缩放到256×192（如下图右侧所示）。

使用上面两者方式的结果才是准确的。<font color='red'><b>如果直接简单粗暴的拉伸目标，准确率会明显下降</b></font>。因为作者源码中训练网络时始终保证目标的比例不变，那么我们在预测时也要保证相同的处理方式，即保证目标比例不变。

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-07-09-22-26-53.png
    width=50%></br><center></center>
</div>

> 💡 其实第二章图片就是YOLOv5中的Letterbox的填充方式，目的就是保持原本的纵横比。

# 前置知识

## MS COCO关键点检测评价指标——OKS<a id=前置知识></a>

链接：[https://cocodataset.org/#keypoints-eval](https://cocodataset.org/#keypoints-eval)

### 1. 关键点评估

本页描述了COCO所使用的*关键点评估指标*。这里提供的评估代码可用于在公开可用的COCO验证集上获得结果。它计算了下面描述的多个指标。要在COCO测试集上获得结果，由于GT是隐藏的，必须将生成的结果上传到评估服务器。测试集上的结果评估使用的是与下面描述完全相同的评估代码。

#### 1.1 评估概述

COCO关键点任务要求同时检测对象并定位它们的关键点（在测试时不提供对象位置）。由于同时检测和关键点估计的任务相对较新，我们选择采用一种受对象检测指标启发的新型度量标准。为了简化，我们将这项任务称为关键点检测，预测算法称为关键点检测器。在继续之前，我们建议先回顾对象检测的评估指标。

评估关键点检测的核心思想是模仿用于对象检测的评估指标，即平均精度（AP）和平均召回率（AR）及其变体。这些指标的核心是<font color='red'><b>真实对象和预测对象之间的相似性度量</b></font>。在对象检测的情况下，IoU作为这种相似性度量（对于框和分割都是如此）。对IoU进行阈值处理定义了真实对象和预测对象之间的匹配，并允许计算精确度-召回率曲线（PR曲线）。为了将AP/AR应用于关键点检测，我们只需要定义一个类似的相似性度量（similarity measure）。我们通过定义对象关键点相似性（object keypoint similarity，OKS）来做到这一点，<font color='blue'><b>它的作用与IoU相同</b></font>。

#### 1.2 对象关键点相似性

对于每个对象，真实关键点具有形式$[x_1,y_1,v_1,...,x_k,y_k,v_k]$，其中$x,y$是关键点位置，$v$是可见性标志，定义为：

$$
\begin{cases}
    v=0:& 未标记 \\
    v=1:& 标记但不可见 \\
    v=2:& 标记且可见
\end{cases}
$$

每个真实对象还有一个比例$s$，我们定义为对象分割区域（segment area）的平方根。

对于每个对象，关键点检测器必须输出关键点位置和对象级别的置信度（keypoint locations and an object-level confidence）。预测的关键点对于一个对象应该具有与GT相同的形式：$[x_1,y_1,v_1,...,x_k,y_k,v_k]$。然而，在评估期间目前不使用检测器预测的$v_i$，即<font color='red'><b>关键点检测器不需要预测每个关键点的可见性或置信度</b></font>。

我们定义对象关键点相似性（OKS）为：

$$
\mathrm{OKS} = \frac{\sum_i [e^{\frac{-d_i^2}{2s^2k_i^2}}\delta(v_i > 0)]}{\sum_i [\delta(v_i > 0)]}
$$

其中，$d_i$ 是每个对应GT关键点和检测到的关键点之间的欧几里得距离，$v_i$ 是GT关键点的可见性标志（检测器预测的 $v_i$ 不使用）。为了计算 OKS，我们将 $d_i$ 通过一个标准差为 $sk_i$ 的未归一化高斯函数，其中 $s$ 是对象的尺度（object scale），$k_i$ 是每个关键点的常数，用于控制下降率。对于每个关键点，这会产生一个介于 $[0, 1]$ 之间的关键点相似度。这些相似度在所有标记的关键点（即 $v_i>0$ 的关键点）上取平均。未标记的预测关键点（$v_i=0$）不影响 OKS。完美的预测将有 $\mathrm{OKS}=1$，而所有关键点都偏离几个标准差 $sk_i$ 的预测将有 OKS 接近 0。OKS 类似于 IoU。有了 OKS，我们可以像 IoU 允许我们计算框/分割检测（box/segment detection）的这些指标一样，计算 AP 和 AR。

#### 1.3 调整 OKS

我们调整 $k_i$，使 OKS 成为一个在感知上有意义且易于解释的相似性度量（a perceptually meaningful and easy to interpret similarity measure）。首先，使用验证集中5000张冗余地标注的图像，对于每种关键点类型 $i$，我们测量了每个关键点相对于对象尺度 $s$ 的标准差 $\sigma_i$。也就是说，我们计算$\sigma_i^2 = \mathrm{E}[\frac{d_i^2}{s^2}]$。$\sigma_i$对于不同的关键点变化很大：人体上的关键点（肩膀、膝盖、臀部等）往往比人头部的关键点（眼睛、鼻子、耳朵）具有更大的$\sigma$。

为了获得一个在感知上有意义且可解释的相似性度量，我们设置 $k_i=2\sigma_i$。在这种 $k_i$ 的设置下，当 $\frac{d_i}{s}$ 的一个、两个和三个标准差时，关键点相似度 $e^{\frac{-d_i^2}{2s^2k_i^2}}$分别取值为 $e^{-1/8}=0.88$、$e^{-4/8}=0.61$ 和 $e^{-9/8}=0.32$。正如预期的那样，人工标注的关键点呈正态分布（忽略偶尔的异常值）。因此，回顾68-95-99.7规则，设置 κi=2σi 意味着 68\%、95\% 和 99.7\% 的人工标注关键点应分别具有 0.88、0.61 或 0.32 或更高的关键点相似度（实际上的百分比为 75\%、95\% 和 98.7\%）。

> 🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：68–95–99.7 rule是什么？
> 🥳 𝑨𝒏𝒔𝒘𝒆𝒓：68-95-99.7规则，也称为经验规则或三西格玛规则，是一个统计规则，它指出在正态分布中，几乎所有观察到的数据都会落在平均值的三个标准差之内（$\sigma$表示标准差，$\mu$表示平均值）。具体来说，这个规则预测，在正态分布中，68\%的观测值会落在第一标准差$\mu + \sigma$之内，95\%会落在前两个标准差$\mu + 2\sigma$之内，而99.7\%会落在前三个标准差$\mu+3\sigma$之内。换句话说，如果数据符合正态分布或近乎正态分布，或者如果你有一个具有单一峰值的对称分布，那么可以使用这个规则来近似估计在一定数量的标准差内会有多少分数。例如，如果一个数据集的平均值为70，标准差为2.5，那么根据经验规则，大约68\%的数据点会落在67.5到72.5的范围内。

OKS 是所有（标记的）对象关键点的平均关键点相似度。下面绘制了预测的 OKS 分布，假设每个对象有 10 个独立的关键点，使用 $k_i=2\sigma_i$（蓝线），以及在双重标注数据上实际的人类 OKS 分数分布（绿线）：

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-07-09-21-49-01.png
    width=50%></br><center></center>
</div>

曲线并不完全匹配有几个原因：（1）对象的关键点不是独立的，（2）每个对象标记的关键点数量不同，以及（3）真实数据包含1-2\%的异常值（其中大部分是由于标注者将左右弄错，或者在两个人靠近时标注错了人）。尽管如此，行为大致符合预期。我们以一些关于人类表现的观察结束：（1）在OKS为0.50时，人类的表现几乎是完美的（95%），（2）人类OKS的中位数约为0.91，（3）当OKS超过0.95后，人类的表现迅速下降。请注意，这个OKS分布可以用来预测人类的AR（因为AR不依赖于误报）。

### 2. 指标

用于描述COCO上关键点检测器性能的以下10个指标：

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-07-09-21-54-10.png
    width=70%></br><center></center>
</div>

1. 除非另有说明，AP（平均精度）和AR（平均召回率）是在多个OKS（对象关键点相似性）值（.50:.05:.95）上平均计算的。
2. 正如所讨论的，我们为每种关键点类型$i$设置了$k_i=2\sigma_i$。对于人体，鼻子、眼睛、耳朵、肩膀、肘部、手腕、臀部、膝盖和脚踝的$\sigma$值分别为0.026、0.025、0.035、0.079、0.072、0.062、0.107、0.087和0.089。
3. 在所有10个OKS阈值上平均计算的AP将决定挑战赛的获胜者。在考虑COCO上关键点性能时，这应该是考虑的最重要的单一指标。
4. 所有指标都是在每张图像最多允许20个高分（top-scoring）检测结果的情况下计算的（我们使用20个检测结果，而不是对象检测挑战赛中的100个，因为目前只有人这一类别有关键点）。
5. 小对象（分割区域\<32<sup>2</sup>）不包含关键点注释。
6. 对于没有标记关键点的对象，包括人群，我们使用一个宽松的启发式方法，允许基于幻想(hallucinated)的关键点（放置在真实对象内，以最大化OKS）匹配检测结果。这与处理带有框/分割的检测中的忽略区域非常相似。
7. 每个对象都被赋予同等的重要性，无论标记/可见的关键点数量如何。我们不会过滤只有少数关键点的对象，也不会根据存在的关键点数量对对象示例进行加权。

# 知识来源

1. [HRNet网络详解](https://www.bilibili.com/video/BV1bB4y1y7qP)
2. [keypoints-eval](https://cocodataset.org/#keypoints-eval)