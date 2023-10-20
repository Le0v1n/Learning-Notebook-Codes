<div align=center>
	<img src=https://img-blog.csdnimg.cn/2cc9035cf2184727b2cb5bb95d184dea.png
	width=100%>
</div>

+ **论文地址**：[PP-LiteSeg: A Superior Real-Time Semantic Segmentation Model](https://arxiv.org/abs/2204.02681)
+ **论文提出时间**：<font color='red'>2022 年 4 月 6 日</font>
+ **PaddlePaddle官方代码**：[https://github.com/PaddlePaddle/PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
+ **Pytorch 复现实现代码**：Coming soon...

---

# 1. PP-LiteSeg 概况

PP-LiteSeg 是一种用于实时语义分割任务的轻量级模型，旨在平衡准确性和速度之间的权衡。PP-LiteSeg 提出了一种灵活轻量的解码器（FLD）来减少以前解码器的计算开销。为了加强特征表示，作者提出了一种统一的注意力融合模块（UAFM），它利用空间和通道注意力产生权重，然后将输入特征与权重融合。此外，作者还提出了一个简单的金字塔池化模块（SPPM），以低计算成本聚合全局上下文。以下是 PP-LiteSeg 模型的主要特点和组成部分：

1. **轻量级模型设计：** PP-LiteSeg 专注于在保持高准确性的同时降低计算开销。这对于实时应用非常重要，因为实时性要求模型能够在短时间内进行推断。

2. **灵活轻量级解码器（FLD，Flexible and Lightweight Decoder，灵活轻量的解码器）：** FLD 是 PP-LiteSeg 的一部分，它被设计用于减少之前解码器的计算负担。通过灵活的结构，FLD 能够在不损失太多性能的情况下减少计算成本。

3. **统一注意力融合模块（UAFM，Unified Attention Fusion Module，统一的注意力融合模块）：** UAFM 是用于增强特征表示的关键组件。它利用空间和通道注意力机制，生成权重，并将输入特征与权重融合。这有助于模型更好地捕获关键的语义信息。

4. **简单金字塔池化模块（SPPM，Simple Pyramid Pooling Module，简单的金字塔池化模块）：** SPPM 的目标是以较低的计算成本汇集全局上下文信息。这有助于提高模型对于环境背景的感知能力，从而增强分割性能。

通过以上设计，PP-LiteSeg 在准确性和速度方面取得了出色的平衡。在 Cityscapes 数据集的测试中，PP-LiteSeg 在 NVIDIA GTX 1080Ti 上实现了 $72.0\%$ 的 mIoU（$273.6  \ \rm FPS$） 和 $77.5\%$ mIoU（$102.6\ \rm FPS$）的优越表现。

# 2. Abstract

> Real-world applications have high demands for semantic segmentation methods. Although semantic segmentation has made remarkable leap-forwards with deep learning, the performance of real-time methods is not satisfactory. In this work, we propose PP-LiteSeg, a novel lightweight model for the real-time semantic segmentation task. Specifically, we present a Flexible and Lightweight Decoder (FLD) to reduce computation overhead of previous decoder. To strengthen feature representations, we propose a Unified Attention Fusion Module (UAFM), which takes advantage of spatial and channel attention to produce a weight and then fuses the input features with the weight. Moreover, a Simple Pyramid Pooling Module (SPPM) is proposed to aggregate global context with low computation cost. Extensive evaluations demonstrate that PP-LiteSeg achieves a superior trade-off between accuracy and speed compared to other methods. On the Cityscapes test set, PP-LiteSeg achieves 72.0% mIoU/273.6 FPS and 77.5% mIoU/102.6 FPS on NVIDIA GTX 1080Ti. Source code and models are available at PaddleSeg: [this https URL](https://github.com/PaddlePaddle/PaddleSeg).

实际世界（Real-world）的应用对于语义分割方法有着较高的要求。<u>尽管深度学习在语义分割方面取得了显著的进展，但实时方法的性能仍然不尽如人意</u>。在这项工作中，我们提出了一种新颖的轻量级模型，名为 PP-LiteSeg，用于实时语义分割任务。具体而言，我们提出了一种灵活轻量级解码器（Flexible and Lightweight Decoder，FLD），以减少先前解码器的计算开销。为了加强特征表示，我们提出了一种统一的注意力融合模块（Unified Attention Fusion Module，UAFM），它利用空间和通道注意力（Spatial and Channel Attention）产生权重，然后将输入特征与权重进行融合。此外，我们还提出了一种简单的金字塔池化模块（Simple Pyramid Pooling Module，SPPM），以较低的计算成本聚合全局上下文。广泛的评估表明，PP-LiteSeg 在准确性和速度之间实现了卓越的权衡。在 Cityscapes 测试集上，PP-LiteSeg 实现了 72.0% 的 mIoU（273.6 FPS） 和 77.5% 的 mIoU（102.6 FPS） 在 NVIDIA GTX 1080Ti 上。源代码和模型可在 [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) 上获得。

> PP-LiteSeg 解决的痛点问题就是：目前语义分割模型在 mIoU 和推理速度很难达到一个 tradeoff，于是提出 PP-LiteSeg 以达到在良好 mIoU 下模型可以有优秀的速度表现 —— 整体思想和 [Mobilenet v3](https://blog.csdn.net/weixin_44878336/article/details/125019271) 很像。
# 3. 网络架构

在语义分割领域，通常采用的是基于深度学习的神经网络模型来实现。这些模型包括了 Encoder、Decoder 以及 Segmentation Head 这些组件，它们各自有着不同的作用，用于将输入图像转换为像素级的语义分割结果。

<div align=center>
	<img src=https://img-blog.csdnimg.cn/2cac00b19e724539be286d2073535a57.png
	width=100%>
</div>

<center><b>图2：架构概述<br>PP-LiteSeg 包括三个模块：编码器（Encoder）、聚合（Aggregation）和解码器（Decoder）</b></center><br>

> 一般而言，Encoder → Decoder 没有聚合的，所以这里我们不进行详细介绍。

- **Encoder（编码器）**：编码器是一个 CNN 的 Backbone，用于从输入图像中提取高级语义特征。它通过一系列卷积层、池化层和激活函数来逐渐减小图像的空间维度 $[H, W]$，并且增强图像中的语义信息（直观体现就是特征图通道数上升）。<font color='red'>编码器的任务是将原始图像转换为具有丰富语义信息的低分辨率特征图</font>，这些特征图包含了图像中的物体、纹理和结构等信息。

- **Decoder（解码器）**：解码器是与编码器相对应的部分，<font color=‘blue’>它负责将编码器产生的低分辨率特征图恢复到原始图像的分辨率，并进行像素级的分类</font>。解码器通常包括上采样层（如反卷积或插值操作）和卷积层，用于逐步恢复特征图的细节，并生成与输入图像大小相同的特征图。解码器的任务是将编码器提取的语义特征映射到像素级别，<font color='red'>以便</font>对每个像素进行分类（<font color='red'>Decoder 本身不会对像素进行分类，而是通过一个分割头来实现</font>）。

- **Segmentation Head（分割头）**：是一个全连接层或卷积层，用于将 Decoder 输出的特征图转换为最终的分割结果（<font color='purple'>将对每一个像素进行分类</font>）。Segmentation Head 的作用是将 Decoder 输出的特征图转换为像素级别的预测结果。

综上所述，在语义分割领域中：
+ 编码器（Encoder）用于提取输入图像的语义信息
+ 解码器（Decoder）用于将语义信息映射回像素级别
+ 分割头（Segmentation）用于最终的像素级别语义分割预测（对每个像素进行最后的分类）

这些组件的协同工作使得深度学习模型能够在像素级别准确地分割出图像中的不同语义区域。接下来我们就对 Encoder 和 Decoder 进行介绍。

## 3.1  【Encoder】STDCNet：更强大的 Backbone

PP-LiteSeg 使用 STDCNet 作为 Backbone。STDCNet 主要有以下优点：

- STDCNet 网络是一种轻量级的卷积神经网络，具有较低的参数量和计算量。它采用了一种名为STDC（Spatial-Temporal Depthwise Convolution）的新型卷积操作，可以有效地提取空间和时间信息，从而提高特征表示能力。
- STDCNet 网络在编码器中使用了多尺度特征融合技术，可以增强编码器对于图像中不同尺度的语义信息的感知能力，提高分割精度。

STDCNet 共有 5 个阶段，每个阶段的步长为 2，因此最终的特征大小是输入图像的 1/32（进行了 32 倍下采样）。基于 STDCNet，作者提出了两种规格的 PP-LiteSeg 网络，如下表所示。

<div align=center>
	<img src=https://img-blog.csdnimg.cn/a154f704a5134e64a7c67dccf9a4b3cd.png
	width=50%>
	<p>表1：不同规格 PP-LiteSeg 的详细信息</p>
</div>

正如表 1 所示，PP-LiteSeg-T 和 PP-LiteSeg-B 的编码器分别是 STDC1 和 STDC2。PP-LiteSeg-B 实现了更高的分割准确性，而 PP-LiteSeg-T 的推断速度更快。

## 3.2 【Aggregation】SPPM（简单金字塔池化模块，Simple Pyramid Pooling Module）

### 3.2.1 SPP 和 PPM 的作用

Encoder 使用 STDCNet 作为 Backbone，对输入图片进行一系列的特征和高级语义提取，之后需要将 Encoder 的输出特征图送入 Decoder 中，在这个过程中，一般会经过一个 MMP 模块（和我们之前讲的 ASPP 作用类似），PPM 主要用于捕获不同尺度上的语义信息，并增强模型对物体在不同尺度下的感知能力，具体作用如下：

- **多尺度特征融合**：在语义分割任务中，不同尺度上的语义信息都是重要的，因为物体的大小和形状可能在图像中有所变化。PPM 通过金字塔池化操作，在不同尺度上对特征图进行池化，然后将池化后的特征进行拼接或融合，从而获得了多尺度的语义信息。这有助于模型更好地理解和分割图像中的不同尺度物体。

- **增强感受野**：PPM 的金字塔池化操作能够有效地扩大神经网络的感受野，使模型能够更好地捕捉图像中的全局和局部信息。通过在不同层级上应用不同大小的池化核，PPM 能够在不丧失分辨率的情况下捕获更广泛的语义信息。

- **提升分割性能**：PPM 可以在解码器的输出特征图上应用，从而为分割任务提供更多的上下文信息。这有助于模型更准确地将像素分类为不同的语义类别，从而提高分割的精度和泛化能力。

---

> **Q**：SPP 和 PPM 是一样的吗？
> **A**：SPP（Spatial Pyramid Pooling）和 PPM（Pyramid Pooling Module）在一定程度上是类似的概念，它们都涉及到对特征图进行金字塔状的池化操作，以捕捉不同尺度上的语义信息。然而，它们在具体实现和应用上存在一些差异。
> SPP 是最早提出的概念，主要应用于图像分类和物体检测任务。它通过对特征图在不同尺度上进行池化，生成固定长度的特征向量，以适应不同尺寸的输入图像。
> PPM 则是在语义分割任务中被引入的，它也采用了类似的金字塔池化策略，但通常更加注重不同尺度的语义信息的融合。PPM 在解码器部分对特征图进行金字塔池化，然后将池化后的特征进行拼接或融合，以获得更全面的语义信息。
> 因此，虽然 SPP 和 PPM 在某种程度上具有相似的思想，但它们的具体设计和应用上存在差异，SPP 更早用于图像分类和物体检测，而 PPM 则更加针对语义分割任务中的特定需求。

---

作者基于 PPM（Pyramid Pooling Module）提出了一个更加简单的 PPM，即 SPPM（Simple Pyramid Pooling Module，SPPM）。如下图所示。

<div align=center>
	<img src=https://img-blog.csdnimg.cn/58b7c3c776a444789989d3b952b1a37d.png
	width=100%>
</div>


### 3.2.2 SPPM 运行流程

1. **利用金字塔池化模块来融合输入特征**：金字塔池化模块包含三个全局平均池化（AvgPool）操作，对应的池化核大小分别为 $1\times 1$、$2\times 2$ 和 $4\times 4$。
2. **输出特征经过 $1 \times 1$ 卷积和上采样操作**：卷积操作使用 $1\times 1$ 的卷积核，输出通道数少于输入通道数（输出特征图的通道减少）。
3. 将这些上采样特征相加，并应用 $3 \times 3$ 卷积操作生成精细特征。

相对于原始的金字塔池化模块（PPM），SPPM 减少了中间和输出通道，去除了原有的拼接操作 `concat`，而是使用了逐元素相加 $\oplus$ 的操作。这个改变旨在降低模型的计算复杂性，使得 SPPM 更加高效，并且适用于实时模型。

### 3.2.3 SPPM 相比原始 SPP 的优点

SPPM相比原始的SPP（Spatial Pyramid Pooling，空间金字塔池化）有以下优点：

- SPPM 只使用了两种不同尺度的池化操作，分别是平均池化（Average Pooling）和最大池化（Max Pooling），而 SPP 使用了多种不同尺度的池化操作（和之前学习的 [ASPP模块](https://blog.csdn.net/weixin_44878336/article/details/132061772) 是一样的），例如 $1\times 1$、$2\times 2$、$3\times 3$、$6\times 6$ 等。这样可以减少池化操作的数量和复杂度，降低计算开销。
- SPPM 将两种不同尺度的池化结果拼接在一起，形成一个多尺度的特征图，而 SPP 将多种不同尺度的池化结果连接在一起，形成一个长向量。这样可以保留更多的空间信息，提高特征表示能力。
- SPPM 在拼接后的特征图上使用了一个 $1\times 1$ 卷积层和一个激活函数，以减少特征图的通道数，并增加非线性变换。这样可以进一步降低计算量，并提高特征表达能力。而 SPP 有使用这样的操作，直接将长向量作为输出。





## 3.3 【Decoder】FLD（Flexible and Lightweight Decoder，灵活轻量级解码器）

### 3.3.1 FLD 设计思想

在语义分割模型中，编码器（Encoder）提取层次化特征，而解码器（Decoder）则融合和上采样特征。对于编码器中从低层到高层的特征，通道数 $[C]$ 增加，空间大小 $[H, W]$ 减小，这是一种高效的设计。

> 网络一般也都是这么设计的，目的是为了减少运算量，提高模型的高级语义信息的提取能力

在一般的语义分割网络中，Decoder 中从高层到低层的特征，空间大小 $[H, W]$ 增加，而通道数 $[C]$ 在最近的模型中保持不变，但是这样会导致一个问题。特征图在 Decoder 中的 $[H, W]$ 在增加，而 $[C]$ 不变，网络的计算开销太大了。

于是，作者提出了 FLD（Flexible and Lightweight Decoder，灵活轻量级解码器），主要作用是<font color='red'>在 Decoder 运行过程中，在逐渐增加特征图的空间大小 $[H, W]$ 的同时，逐渐减少特征的通道数</font>。这种设计平衡了 Encoder 和 Decoder 的计算复杂性，使整体模型更加高效。FLD 架构如下图所示。

<div align=center>
	<img src=https://img-blog.csdnimg.cn/6227b0996e304b2fb53b4d11fa1afd2a.png
	width=100%>
</div>

### 3.3.2 FLD 的优缺点

FLD（Flexible and Lightweight Decoder）相比传统的 Encoder-Decoder 结构的优缺点如下：

+ 【**优点**】FLD 可以逐渐减少解码器中的通道数量，增加特征图的空间大小，从而减少解码器的冗余性，并平衡编码器和解码器的计算成本。
+ 【**缺点**】这种结构可能会降低分割精度，因为解码器中的特征图通道数较低，可能会丢失一些高级语义信息。此外，FLD 需要更多的超参数调整和实验验证，以找到最佳的结构和参数组合。


## 3.4 【Decoder】UAFM（统一注意力融合模块，Unified Attention Fusion Module）

之前不管是学习 FCN 还是 DeepLab 系列模型，我们都知道一个道理 —— **融合多级特征对于实现高分割准确性是至关重要的**。

作者基于此提出了一个统一注意力融合模块 (UAFM)，它利用空间和通道注意力机制（两种注意力机制），生成一个权重矩阵 $\alpha$，并将输入特征与权重矩阵相乘，得到融合后的特征。这样可以使模型更好地捕获图像中的关键语义信息，提高分割精度。UAFM 如下图所示。

![在这里插入图片描述]( =x400)

<div align=center>
	<img src=https://img-blog.csdnimg.cn/5eea7267ad444040bcbf378f404ec52b.png
	width=100%>
</div>

如上图所示，UAFM 利用注意力模块生成权重 $\alpha$，并通过 Mul 和 Add 操作将输入特征与 $\alpha$ 融合。具体而言，输入特征被表示为 $F_{high}$ 和 $F_{low}$。$F_{high}$ 是深层模块的输出，$F_{low}$ 是来自编码器的对应部分。需要注意的是，它们具有相同的通道数。

> **注意**：
> + 注意力模块可以是空间注意力或者是通道注意力，看具体任务要求定（作者在实验的时候使用的是空间注意力，说是为了提高模型运行速度）。
> + 之前的低级语义信息的特征图和高级语义信息的特征图融合可能就是直接 $\oplus$，这样非常粗暴，作者就想改善这个过程，因此就在 $\oplus$ 之前添加了一个注意力模块，让注意力模块可以计算出两个特征图的权重，之后再相加后就比直接 $\oplus$ 效果好了。

UAFM 首先使用双线性插值操作将 $F_{high}$ 上采样到与 $F_{low}$ 相同的大小，上采样后的特征表示为 $F_{up}$。然后，注意力模块将 $F_{up}$ 和 $F_{low}$ 作为输入，生成权重 $\alpha$。之后，为了获得注意力加权特征，我们分别对 $F_{up}$ 和 $F_{low}$ 进行逐元素乘法操作 $\otimes$。最后，UAFM 对注意力加权特征进行逐元素加法操作 $\oplus$，输出融合特征。我们可以将上述过程表示为公式 1。

$$
\begin{aligned}
F_{up} & = \mathrm{Upsample}(F_{high})\\
& \alpha=\mathrm{Attention}(F_{up}, F_{high})\\
F_{out} & = \alpha \cdot F_{up} + (1 - \alpha) \cdot F_{low} \tag{1}
\end{aligned}
$$

### 3.4.1 空间注意力模块

空间注意力模块的动机是利用像素之间的空间关系产生一个权重，该权重表示输入特征中每个像素的重要性，如图4 (a) 所示。

<div align=center>
	<img src=https://img-blog.csdnimg.cn/4bf37a855c3c43958b551cda0dc83928.png
	width=100%>
	<p>图 4：(a) 空间注意力模块</p>
</div>

给定输入特征，即 $F_{up} \in \mathbb{R}^{C\times H\times W}$ 和 $F_{low} \in \mathbb{R}^{C\times H\times W}$，我们首先沿着通道轴执行均值（Mean）和最大（Max）操作，生成四个特征，其维度为 $\mathbb{R}^{1\times H\times W}$。然后，这四个特征被连接成一个特征 $F_{cat} \in \mathbb{R}^{4\times H\times W}$。对于连接的特征，我们应用卷积（Conv）和 Sigmoid 操作，输出 $\alpha \in \mathbb{R}^{1\times H\times W}$。空间注意力模块的公式表示如公式 2 所示。

$$
\begin{aligned}
F_{cat} & = \mathrm{Concat}[\mathrm{Mean}(F_{up}), \mathrm{Max}(F_{up}), 
			        							\mathrm{Mean}(F_{low}), \mathrm{Max}(F_{low})]\\
& \alpha=\mathrm{Sigmoid}(\mathrm{Conv}(F_{cat})) \tag{2}
\end{aligned}
$$

### 3.4.2 通道注意力模块

通道注意力模块的关键是利用通道之间的关系生成权重，该权重指示了输入特征中每个通道的重要性，如图 4 (b) 所示。

<div align=center>
	<img src=https://img-blog.csdnimg.cn/b22578ac39704e008bb1dc6be1f1581a.png
	width=100%>
	<p>图 4：(b) 通道注意力模块</p>
</div>

通道注意力模块首先利用平均池化（AvgPool）和最大池化（MaxPool）操作来压缩输入特征的空间维度。这个过程生成了维度为 $\mathbb{R}^{C\times 1\times 1}$ 的四个特征。然后，将这四个特征沿着通道轴进行连接，然后执行卷积（Conv）和 Sigmoid 操作来产生一个权重 $\alpha \in \mathbb{R}^{C\times 1\times 1}$。简而言之，通道注意力模块的过程可以表示为公式 3。

$$
\begin{aligned}
F_{cat} & = \mathrm{Concat}[\mathrm{AvgPool}(F_{up}), \mathrm{MaxPool}(F_{up}), 
			        							\mathrm{AvgPool}(F_{low}), \mathrm{MaxPool}(F_{low})]\\
& \alpha=\mathrm{Sigmoid}(\mathrm{Conv}(F_{cat})) \tag{3}
\end{aligned}
$$

## 3.5 PP-LiteSeg 整体框架

前 4 小节中对 PP-LiteSeg 中的核心组件进行了介绍，那么 PP-LiteSeg 的基本原理我们已经明白了，下面看一下 PP-LiteSeg 的整体框架。

<div align=center>
	<img src=https://img-blog.csdnimg.cn/da3fd1295f9a41dfacf747ab7c709ba2.png
	width=100%>
</div>


**主要注意的点**：

1. 因为 PP-LiteSeg 使用了 STDCNet 作为 Backbone，因此作者在训练的时候使用了该模型的预训练权重，从而实现加速训练的效果（预训练模型非常重要，我们一般都是会用的，使用前后准确率相差很大，可以参考我之前做的实验 → [FCN代码及效果展示](https://blog.csdn.net/weixin_44878336/article/details/128437026)）。
2. FLD 包含两个 UAFM 和一个分割头。为了提高运行效率，**UAFM 中只采用了空间注意力模块**。最后一个 UAFM 会对输入进行 8 倍下采样。
3.  Segmentation Head 会先执行 `Conv-BN-ReLU` 操作将 进行了 8 倍通道（Channel）下采样的特征的通道数减少到类别数（`num_classes`）。接下来进行上采样操作，将特征大小扩展到输入图像大小，并进行 argmax 操作预测每个像素的标签。
4. 采用了带有在线难例挖掘的交叉熵（The cross entropy loss with Online Hard Example Mining）作为损失函数来进行模型参数优化

> 在线难例挖掘的交叉熵损失（The cross entropy loss with Online Hard Example Mining）是一种训练深度学习模型的损失函数，在处理具有类别不平衡或者难以分类的样本时特别有效。它结合了交叉熵损失和难例挖掘的思想。
交叉熵损失通常用于分类问题中，用于衡量模型预测的类别分布与真实标签的分布之间的差异。然而，当存在类别不平衡或一些样本难以分类时，普通的交叉熵损失可能会导致模型过于关注容易分类的样本，而忽略了那些难以分类的样本。
在线难例挖掘的交叉熵损失通过选择一些难以分类的样本，将它们的损失权重增加，从而强制模型更多地关注这些难例。具体做法是在每个训练批次中，计算损失后对样本进行排序，然后选择一定数量的难例样本（如前一部分或一定百分比的样本），并为它们分配更高的损失权重。这样可以促使模型更好地适应那些难以分类的情况，提高整体性能。
总之，在线难例挖掘的交叉熵损失是一种用于处理难例和类别不平衡的训练策略，有助于提升模型在困难样本上的表现。

# 4. 实验部分

因为 PP-LiteSeg 主打的就是一个速度和 meanIoU 的 trade-off，因此作者在实验部分在准确性和推断速度方面与其他 SOTA 的实时（Real-Time）方法进行实验结果的比较。

## 4.1 数据集介绍

论文使用了两个数据集，分别是 Cityscapes 和 CamVid 数据集，下面我们对这两者数据集进行简单地介绍。

当然，我很乐意为您介绍Cityscapes和CamVid数据集。

### 4.1.1 Cityscapes 数据集

Cityscapes 数据集是一个用于计算机视觉任务的大规模 <font color='red'>城市场景</font> 分割数据集。它主要用于分割任务，即将图像中的每个像素分配到特定的语义类别，如道路、建筑物、行人等。以下是有关 Cityscapes 数据集的一些关键信息：

- **图像数量：** 数据集包含来自德国和其他欧洲城市的大约 5000 张高分辨率图像。
- **类别数量：** 数据集中共有30个不同的语义类别，包括道路、建筑物、车辆、行人等。
- **图像分辨率：** 图像分辨率较高，通常为 $1024\times 2048$ 像素。
- **标签信息：** 每个像素都被标记为特定的语义类别，用于训练和评估分割模型。



### 4.1.2 CamVid 数据集

CamVid 数据集是另一个常用于语义分割任务的数据集，它也包含 <font color='red'>城市场景图像</font> 及其像素级标签。以下是关于CamVid数据集的一些要点：

- **图像数量：** CamVid 数据集包含 701 张图像，这些图像也来自城市环境。
- **类别数量：** 数据集涵盖了 11 个不同的语义类别，包括道路、行人、车辆等。
- **图像分辨率：** 图像分辨率通常为 $960 \times 720$ 像素。
- **标签信息：** 同样地，每个像素都被标记为其对应的语义类别。

> 虽然 CamVid 数据集规模较小，但它仍然在语义分割算法的开发和评估中具有一定的价值。<font color='green'>由于其规模较小，它常常被用作一种小型的基准数据集，用于快速验证分割模型的性能</font>。

## 4.2 训练设置

<div align=center>

|Item|通用配置 1|通用配置 2|Cityscapes|CamVid|
|:-|:-:|:-:|:-:|:-:|
|SGD 优化器|`momentum=0.9`|`lr=poly(lr)`|`weight_decay=5e-4`|`weight_decay=1e-4`|
|预测策略|warm-up||||
|Batch size|||`16`|`24`|
|Iterations|||`160,000`|`1,000`|
|初始学习率|||`0.005`|`0.01`|
|||
|随机缩放|||$[0.125, 1.5]$|$[0.5, 2.5]$|
|Crop size|||$1024×512$|$960 × 720$|
|服务器|NVIDIA Tesla V100||||
|平台|PaddlePaddle v1||||
|模型代码|[PaddleSeg2](https://github.com/PaddlePaddle/PaddleSeg)||||

</div>

**Q**：[为什么 PaddleSeg 不采用设置 epoch 的方式？](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/docs/faq/faq/faq_cn.md#q2-%E4%B8%BA%E4%BB%80%E4%B9%88paddleseg%E4%B8%8D%E9%87%87%E7%94%A8%E8%AE%BE%E7%BD%AEepoch%E7%9A%84%E6%96%B9%E5%BC%8F)
**A**：设置 epoch 的方式会受数据集大小的影响。因此PaddleSeg 按照 iters 进行设置。

+ 数据集大小： $N$
+ 批量大小：$\mathrm{batch_size}$
+ GPU数量： $\rm num_gpus$
+ 总迭代次数： $\rm iters$

则有：$$\rm epoch = \frac{iters * batch\_size \times num\_gpus}{N}$$

## 4.3 推理设置

<div align=center>

|Item|通用配置 1|通用配置 2|Cityscapes|CamVid|
|:-|:-:|:-:|:-:|:-:|
|推理模型格式|[ONNX](https://github.com/microsoft/onnxruntime)|[TensorRT](https://github.com/NVIDIA/TensorRT)|||
|推理时间|||Crop($1024 × 512$ & $1536 × 768$) → Infer → Resize(原图大小)|Crop($960×720$) → Infer|
|推理 GPU|NVIDIA 1080Ti||||
|环境配置|CUDA 10.2 + CUDNN 7.6|TensorRT 7.1.3|||
|推理耗时单位|FPS||||
|准确性单位|mean IoU||||

</div>

> 为了进行公平比较，将 PP-LiteSeg 导出为 ONNX 格式，并利用 TensorRT 执行模型。与其他方法类似，首先将 Cityscapes 数据集中的图像缩放到 $1024 × 512$ 和 $1536 × 768$，然后推断模型将缩放后的图像作为输入，生成预测图像，最后将预测图像重新调整到原始输入图像的大小。这三个步骤的耗时被计算为推断时间。对于 CamVid 数据集，推断模型以原始图像作为输入，分辨率为 $960×720$。在 NVIDIA 1080Ti GPU 上使用 CUDA 10.2、CUDNN 7.6 和 TensorRT 7.1.3 进行所有推断实验。使用标准的平均交并比（mIoU）进行分割准确性比较，使用帧率（FPS）进行推断速度比较。

## 4.4 Cityscapes 数据集的结果

### 4.4.1 与 SOTA 方法对比

在上述的训练和推断设置下，我们在 Cityscapes 数据集上将提出的 PP-LiteSeg 与之前的 SOTA 实时模型进行了比较。为了公平比较，我们在两个不同分辨率下评估了 PP-LiteSeg-T 和 PP-LiteSeg-B，即 $512 × 1024$ 和 $768 × 1536$。表 2 展示了各种方法的模型信息、输入分辨率、mIoU 和 FPS。图 1 提供了分割准确性和推断速度的直观比较。


<div align=center>
	<img src=https://img-blog.csdnimg.cn/b1a0822c6d0546bba0f0a3f1a4e9df82.png
	width=65%>
	<p>表 2：在 Cityscapes 数据集上与 SOTA 实时方法的比较</p>
</div>

从表 2 中可以看到， PP-LiteSeg-B2 在 Cityscapes 验证集和测试集有最高的 mIoU，且速度处于中等位置，因为 B2 模型本身比较大，所以取得了良好的准确率，但在速度上没有达到一个非常好的 tradeoff；PP-LiteSeg-T1 是最小的模型，有最快的速度，且 mIoU 相对处于中等的位置，因此整体表现不错。

说实话，从一堆数据中找到规律其实是不直观的，我们可以看一下下面这个图，可以直观地感受到 PP-LiteSeg 在 mIoU 和速度上的优势。

<div align=center>
	<img src=https://img-blog.csdnimg.cn/58ca186191a1488d9004dfbb102ad389.png
	width=65%>
	<p>图1：在 Cityscapes 测试集上分割准确性（mIoU）和推理速度（FPS）的比较</p>
</div>

结合两张图我们可以看到，PP-LiteSeg 在准确性和速度之间达到了 SOTA 的平衡。具体来说，我们可以看到 PP-LiteSeg-T1 实现了 273.6 FPS 和 72.0% 的 mIoU，这意味着最快的推断速度和具有竞争力的准确性。在分辨率为 $768 × 1536$ 的情况下，PP-LiteSeg-B2 在验证集上获得了最佳准确性，即 78.2% 的 mIoU，测试集上为 77.5% 的 mIoU。此外，与 STDC-Seg 使用相同的编码器和输入分辨率相比，PP-LiteSeg 表现得更好。

### 4.4.2 消融实验

作者也进行了消融实验来证明提出的模块的有效性。实验选择了 PP-LiteSeg-B2 模型（mIoU 最高的模型），并采用了相同的训练和推断设置。Baseline Model 是没有使用提出的模块的 PP-LiteSeg-B2，其中解码器中的特征通道数为 96，并且融合方法是逐元素相加 $\oplus$。表格 3 展示了消融实验的定量结果。

<div align=center>
	<img src=https://img-blog.csdnimg.cn/fae9f4a673a04312a3c6d340ca859be0.png
	width=55%>
	<p>表 3：在 Cityscapes 验证集上对我们提出的模块进行的消融实验<br>Baseline Model 是没有使用提出的模块的 PP-LiteSeg-B2
	</p>
</div>

可以看到，在 PP-LiteSeg-B2 模型中，引入 FLD 模块将 mIoU 提升了 0.17%。引入 SPPM 和 UAFM 也提高了分割准确性，尽管推断速度稍微降低。基于三个提出的模块，PP-LiteSeg-B2 在 102.6 FPS 的情况下实现了 78.21 的 mIoU。与 Baseline 模型相比，mIoU 提高了 0.71%。图 6 提供了定性比较。

<div align=center>
	<img src=https://img-blog.csdnimg.cn/62755b1ba6bc4f05981917a8db1cece2.png
	width=100%>
	<p>图 6：在 Cityscapes 验证集上的定性比较<br>(a)-(e) 分别代表Baseline、Baseline + FLD、Baseline + FLD + SPPM、Baseline + FLD + UAFM 和<br>Baseline + FLD + SPPM + UAFM 的预测图像，(f) 代表真实标签
	</p>
</div>

从上图可以观察到，随着逐个添加 FLD、SPPM 和 UAFM，预测图像与真实标签之间的一致性更加明显。总之，我们提出的模块对于语义分割是有效的。

## 4.5 CamVid 数据集的结果

为了进一步展示 PP-LiteSeg 的能力，作者还在 CamVid 数据集上进行了实验。与其他工作类似，训练和推断的输入分辨率为 $960×720$。

<div align=center>
	<img src=https://img-blog.csdnimg.cn/42cc10154d014550acef69e70842a5f2.png
	width=50%>
	<p>表 4：在 CamVid 测试集上与最先进的实时方法的比较<br>所有方法的输入分辨率为 960×720
	</p>
</div>

如表 4 所示，PP-LiteSeg-T 实现了 222.3 FPS 的速度，比其他方法快了 12.5% 以上。PP-LiteSeg-B 实现了最佳的准确率，即 75.0% 的 mIoU，速度为 154.8 FPS。总体来说，比较结果显示 PP-LiteSeg 在 CamVid 数据集上在准确率和速度之间达到了最佳平衡。

# 5. 总结

作者针对实时语义分割（Real-Time Semantic Segmentation）设计了一种新颖的网络 PP-LiteSeg，并提出一系列 tricks：

1. 提出了 FLD（灵活轻量级解码器）：提高以前解码器的效率。
2. 提出了 UAFM （统一的注意力融合模块）：使用注意力模块增强浅层和深层特征图的融合效果
3. 提出了 SPPM（简单的金字塔池化模块）：相比 PPM，以较低计算成本聚合全局上下文。

基于上面这些新颖的模块，作者提出了 PP-LiteSeg 语义分割模型。在 Cityscapes 和 CamVid 数据集上大量的实验结果表明，PP-LiteSeg 在分割准确性和推理速度之间取得了最先进的平衡（trade-off）。