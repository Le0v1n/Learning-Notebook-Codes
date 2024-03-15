
<div align=center>
    <img src=./imgs_markdown/2024-03-14-19-41-35.png
    width=100%>
    <center></center>
</div>

- **论文地址**：[Fast-SCNN: Fast Semantic Segmentation Network](https://arxiv.org/abs/1902.04502)
- **论文提出时间**：<font color='red'>2019 年 2 月 12 日</font>
- **官方代码**：作者并没有放出源代码，因此下面是一些第三方的实现
  - [PaddleSeg 复现代码](https://github.com/PaddlePaddle/PaddleSeg)：百度飞桨团队复现的代码
  - [MMSegmentation 复现代码](https://github.com/open-mmlab/mmsegmentation)：商汤团队复现的代码
  - [他人复现的 PyTorch 代码](https://github.com/Tramac/Fast-SCNN-pytorch)：并非 PyTorch 官方复现的代码
  - [更多代码](https://paperswithcode.com/paper/fast-scnn-fast-semantic-segmentation-network)

> 虽然没有源代码，但是有很多第三方对其进行了复现，说明该模型还是有效果的 :joy: 

# Abstract

编解码器（encoder-decoder）框架是离线语义图像分割（offline semantic image segmentation）的最新技术。随着自主系统（autonomous systems）的兴起，实时计算（real-time computation）变得越来越受欢迎。在本文中，我们介绍了快速分割卷积神经网络（Fast-SCNN），这是一种针对高分辨率图像数据（1024x2048 像素）的**实时语义分割模型，适合在内存较低的嵌入式设备上进行高效计算**。基于现有的快速分割双分支方法（two-branch methods），我们引入了我们的“学习下采样（learning to downsample）”模块，该模块同时为多个分辨率分支计算低级特征。我们的网络结合了高分辨率下的空间细节和低分辨率下提取的深层特征，在 Cityscapes 数据集上实现了平均交并比（mean intersection over union，mIoU）为 68.0%，每秒 123.5 帧的速度。我们还展示了大规模预训练是不必要的。我们在使用 ImageNet 预训练和 Cityscapes 的粗略标注数据的实验中彻底验证了我们的指标。最后，我们展示了在子采样输入（subsampled inputs）上的更快计算，且无需任何网络修改即可获得具有竞争力的结果。

---

<kbd><b>Question</b></kbd>：什么是离线语义图像分割（offline semantic image segmentation）

<kbd><b>Answer</b></kbd>：简单理解，“离线语义图像分割” <=> “离线语义分割”。那么关键点就在于**离线**。离线的反义词是“实时”，意思就是说，实时语义分割对模型性能要求比较高，模型应该有较小的参数来保证运行效率。对于离线语义分割而言，模型的大小无所谓，速度并不是考虑的因素，因为输入图片可能非常大，比如遥感图像、医学图像，这些任务对模型速度要求不高，但对精度要求高。

---

<kbd><b>Question</b></kbd>：什么是自主系统（autonomous systems）？

<kbd><b>Answer</b></kbd>：自主系统（Autonomous Systems）是能够独立执行任务和做出决策的系统。它们不需要人类的干预，通过感知环境、分析数据和学习能力来做出决策和行动。自主系统包括机器人、自动驾驶车辆、智能无人机等，具备感知、决策、控制和学习能力。它们的目标是提高效率、减少错误和风险，并为人类创造更舒适和安全的生活环境。

> 结合该论文使用的数据集 CityScapes 可以推断出，这里其实主要说的是自动驾驶场景。

---

<kbd><b>Question</b></kbd>：在语义分割中，快速分割双分支方法（two-branch methods）是什么？

<kbd><b>Answer</b></kbd>：快速分割双分支方法是一种用于高效语义分割的技术。它通过同时进行像素级别的语义分割和全局语义分割来提高分割速度。这种方法使用两个分支：

- **语义分割分支**：负责像素级别的分割
- **全局语义分割分支**：提供整体语义信息的上下文

通过结合这两个分支，快速分割双分支方法在保持较高分割精度的同时加快了分割速度。

---

<kbd><b>Question</b></kbd>：什么是子采样输入（subsampled inputs）？

<kbd><b>Answer</b></kbd>：子采样输入是指对图像进行**降采样**或**缩小尺寸**的操作。它可以减少计算量、存储需求，并提取重要特征。通过降低图像的分辨率或缩小尺寸，子采样输入**能够在保持关键信息的同时提高处理效率**。

---

<kbd><b>Question</b></kbd>：CityScapes 数据集是什么？主要用于哪个领域？

<kbd><b>Answer</b></kbd>：Cityscapes 数据集是一个用于计算机视觉领域的开放数据集，主要用于语义分割任务。它包含高分辨率的城市街道图像，用于训练和评估语义分割算法。Cityscapes 数据集在城市场景理解、自动驾驶和智能交通系统等领域有广泛应用。

> CityScapes 数据集官网链接：[The Cityscapes Dataset
Semantic, instance-wise, dense pixel annotations of 30 classes](https://www.cityscapes-dataset.com/)

# 1. Introduction

快速语义分割在实时应用中非常重要，这些应用需要快速解析输入数据，以实现与环境的快速交互。由于对自主系统和机器人的兴趣日益增长，因此可以明显看出，实时语义分割的研究近年来受到了显著的关注和发展。作者强调**超越实时性能通常是必要的**，因为语义标签通常仅用作其他时间关键任务的预处理步骤（传说中的组合拳🤣）。此外，嵌入式设备上的实时语义分割（无需强大的 GPU 支持）可能会实现许多其他应用，例如可穿戴设备上的增强现实技术。

作者发现，在解决语义分割问题时，通常采用深度卷积神经网络（DCNN）的编码器-解码器框架，而许多高效运行时的实现则采用了两个或多个分支的架构。在这种情况下：

- 学习对象类之间复杂关联（即全局上下文）时，**较大的感受野**非常重要
- 为了保留对象边界，图像中的**空间细节**是必不可少的
- 需要进行特定设计，以在速度和准确性之间取得平衡，而不是简单地重新 DCNN（即重新设计 backbone）

在具有两个分支的网络中，我们采用了一种更深的分支来捕捉低分辨率（下采样倍率高的特征图）下的全局上下文，同时设置了一个浅层分支（下采样率倍率没有前者那么高）来学习完整输入分辨率下的空间细节。最终的语义分割结果通过合并这两个分支得到。需要特别注意的是，由于较小的输入尺寸克服了更深网络的计算成本，而且只有在少数层次上使用完整分辨率进行执行，因此现代 GPU 可以实现实时性能。与编码器-解码器框架不同的是，在两个分支方法中，不同分辨率下的初始卷积是不共享的。值得一提的是，引导式上采样网络（GUN）和图像级联网络（ICNet）只在前几层之间共享权重，而计算过程并不共享。

在这项工作中，我们提出了一种名为 Fast-SCNN 的快速分割卷积神经网络，它是一种超越实时的语义分割算法，将先前的两个分支设置与经典的编码器-解码器框架相结合（如[图 1](#fig1) 所示）。

<div align=center>
    <img src=./imgs_markdown/2024-03-14-20-21-55.png
    width=100%><a id='fig1'></a>
    <center>图 1. Fast-SCNN 通过在两个分支（编码器）之间共享计算来构建一个超越实时的语义分割网络</center></br>
</div>

基于我们观察到的初级 DCNN 层提取低级特征的现象，我们在两个分支方法中共享了初始层的计算。我们将这种技术称为学习下采样。这种效果类似于编码器-解码器模型中的跳跃连接，但我们只使用一次跳跃连接以保持运行时效率，并且保持模块的浅层结构以确保特征共享的有效性。最后，我们的 Fast-SCNN 采用了高效的深度可分离卷积和逆残差块（出自 [MobileNetV2](https://blog.csdn.net/weixin_44878336/article/details/125019271)）。

在 Cityscapes 数据集上应用 Fast-SCNN，使用全分辨率（1024×2048 像素），在 Nvidia Titan Xp (Pascal) 上以每秒 123.5 fps的速度，获得了 68.0% 的 mIoU。这比之前的方法 BiSeNet（71.4% mIoU）快了两倍。

虽然 Fast-SCNN 使用了 111 万个参数，但大多数离线分割方法（例如 [DeepLabV1](https://blog.csdn.net/weixin_44878336/article/details/131961813) 和 PSPNet ）以及一些实时算法（例如 GUN 和 ICNet）所需的参数要比这个多得多。

Fast-SCNN 的模型容量被特意保持较低。原因有两个：

1. 较低的内存消耗使其能够在嵌入式设备上运行
2. 期望获得更好的泛化性能

> 💡  **Tips**：在数据集不变的情况下，模型参数量越大越容易过拟合。所以模型变大了，数据集也应该变大。

很多人在论文中经常建议模型在 ImageNet 上进行预训练，从而提高准确性和泛化性能。作者不信邪，研究了预训练对低容量 Fast-SCNN 的影响。与高容量网络的趋势相反，作者发现预训练或额外的粗标注训练数据对结果的影响微乎其微。具体来说，**在 Cityscapes 数据集上，使用预训练权重和额外的粗糙数据的轻量版 Fast-SCNN 的 mIoU 只提升了 0.5%**。

> 💡  **Tips**：
> - 低容量和高容量：不同参数量的模型
> - 粗标注训练数据：标注没有那么精良的数据，即相对粗糙的数据

总结起来，作者的贡献包括：

1. 提出了 Fast-SCNN，这是一种适用于高分辨率图像（1024×2048 像素）的竞争性（68.0%）和超越实时的语义分割算法（123.5 fps）。

2. Fast-SCNN 采用了离线 DCNN 中常用的跳跃连接（shortcut），并提出了一个浅层的学习下采样（learning down-sample）模块，用于快速高效地进行多分支低级特征提取。

3. 特意设计了低容量的 Fast-SCNN，并通过实验证明，在小容量的 Fast-SCNN 模型中，运行更多的训练 Epoch 与在 ImageNet 上进行预训练或使用额外的粗糙数据进行训练具有相同的成功效果。

4. 作者将 Fast-SCNN 应用于子采样输入数据，无需重新设计网络即可实现最先进的性能。

# 2. 相关工作

## 2.1 语义分割发展

最先进的语义分割深度卷积神经网络（DCNNs）结合了两个独立的模块：编码器（Encoder）和解码器（Decoder）。编码器模块使用卷积和池化操作的组合来提取 DCNN 特征。解码器模块从子分辨率特征中恢复空间细节，并预测对象的标签（即语义分割）。通常情况下，编码器是从简单的分类 DCNN 方法（如 VGG 或 ResNet ）进行调整的。在语义分割中，全连接层（Fully Connected Layer）被移除。

开创性的全卷积网络（FCN）为大多数现代分割架构奠定了基础。具体而言，FCN 使用 VGG 作为编码器，并结合双线性上采样和来自较低层的跳跃连接来恢复空间细节。U-Net 进一步利用密集的跳跃连接来提取空间细节。

> 关于 FCN 这个经典网络的介绍，可以看下面的博客：
> - [FCN理论部分](https://blog.csdn.net/weixin_44878336/article/details/126343707)
> - [FCN代码及效果展示](https://blog.csdn.net/weixin_44878336/article/details/128437026)

后来，受到全局图像级上下文在 DCNNs 之前的启发，PSPNet 的金字塔池化模块和 DeepLab 的空洞空间金字塔池化（ASPP）被用于编码和利用全局上下文。

> 关于 DeepLab 这个经典网络的介绍，可以看下面的博客：
> - [DeepLab v1网络](https://blog.csdn.net/weixin_44878336/article/details/131961813)
> 
> 关于金字塔池化，可以看这篇博客：
> - [ASPP不同版本对比（DeepLab、DeepLab v1、DeepLab v2、DeepLab v3、DeepLab v3+、LR-ASPP）](https://blog.csdn.net/weixin_44878336/article/details/132061772)

其他竞争性的基本分割架构使用条件随机场（CRF）或递归神经网络。然而，它们都不能实时运行。

与目标检测类似，速度成为图像分割系统设计中的一个重要因素。在 FCN 的基础上，SegNet 引入了一个联合编码器-解码器模型，成为最早的高效分割模型之一。在 SegNet 之后，ENet 也设计了一个具有少量层的编码器-解码器，以降低计算成本。

最近，引入了两个分支和多个分支的系统。ICNet、ContextNet、BiSeNet 和 GUN 在深层分支中学习降低分辨率输入的全局上下文，而在浅层分支中学习边界的全分辨率。

然而，最先进的实时语义分割仍然具有挑战性，并且通常需要高端的 GPU。受到两个分支方法的启发，Fast-SCNN 引入了一个共享的浅层网络路径来编码细节，同时在低分辨率下高效地学习上下文（如[图 2](#fig2) 所示）。

<div align=center>
    <img src=./imgs_markdown/2024-03-15-09-25-46.png
    width=80%><a id='fig2'></a>
    <center>图 2. Fast-SCNN 与编码器-解码器和两个分支架构的示意比较。编码器-解码器在多个分辨率上使用多个跳跃连接，通常是由深度卷积块产生的。两个分支方法利用低分辨率的全局特征和浅层空间细节。Fast-SCNN 在我们的学习下采样模块中同时编码空间细节和全局上下文的初始层。</center></br>
</div>

> 第一种就是 FCN 那样的思想，从浅层得到特征与深层进行融合。第二种我没有见过，但是整体结构也比较清晰，输入有两个，一个正常尺寸的图像，一个小图，最后再将二者的结果进行融合。第三种就是今天的主角，Fast-SCNN 提出来的，但这样和 FCN 有什么区别呢？我们继续往下看。

## 2.2 DCNNs 的速度

DCNNs 的高效技术通常可以分为四类：

1. **深度可分离卷积**：MobileNet 将标准卷积分解为深度卷积和 1×1 逐点卷积，称为深度可分离卷积。这种分解减少了浮点运算和卷积参数，从而降低了模型的计算成本和内存需求。

2. **DCNN 的高效重新设计**：Chollet 使用高效的深度可分离卷积设计了 Xception 网络。MobileNet-V2 提出了倒置瓶颈残差块，用于构建用于分类任务的高效 DCNN。ContextNet  使用倒置瓶颈残差块设计了一个用于高效实时语义分割的两个分支网络。

3. **网络量化**：由于浮点乘法比整数或二进制运算昂贵，可以使用量化技术对 DCNN 的滤波器和激活值进行量化，从而进一步减少运行时间。

4. **网络压缩**：剪枝可以应用于减小预训练网络的大小，从而实现更快的运行时间、更小的参数集和更小的内存占用。

Fast-SCNN 在很大程度上依赖于深度可分离卷积和残差瓶颈块。此外，我们引入了一个两个分支的模型，其中包含我们的学习下采样模块，允许在多个分辨率级别上共享特征提取（如[图 2](#fig2) 所示）。需要注意的是，即使多个分支的初始层提取了类似的特征，常见的两个分支方法也没有利用这一点。网络量化和网络压缩可以同时应用，这留待将来的工作。

## 2.3. 辅助任务的预训练

有一种普遍的观点认为，在辅助任务上进行预训练可以提高系统的准确性。早期在目标检测和语义分割方面的研究表明，通过在 ImageNet 上进行预训练可以实现这一点。随着这一趋势，其他实时高效的语义分割方法也在 ImageNet 上进行了预训练。然而，目前尚不清楚在低容量网络上是否需要进行预训练。Fast-SCNN 专门设计为低容量网络。在我们的实验中，我们展示了小型网络从预训练中并没有获得显著的好处。相反，积极的数据增强和更多的训练轮次可以提供类似的结果。

# 3. Fast-SCNN

Fast-SCNN 受到了两个分支架构和具有跳跃连接的编码器-解码器网络的启发。我们注意到早期层通常提取低级特征。我们重新解释跳跃连接作为一个学习下采样模块，使我们能够融合这两种框架的关键思想，并构建一个快速的语义分割模型。[图 1](#fig1) 和[表 1](#table1) 展示了 Fast-SCNN 的布局。接下来，我们将讨论我们的动机，并更详细地描述我们的构建模块。

<div align=center>
    <img src=./imgs_markdown/2024-03-15-09-40-12.png
    width=60%><a id='table1'></a>
    <center>表 1. Fast-SCNN 使用标准卷积(Conv2D)、深度可分离卷积(DSConv)、反向残差瓶颈块(bottleneck)、金字塔池化模块(PPM)和特征融合模块(FFM)。参数 t、c、n 和 s 分别表示瓶颈块的扩展因子、输出通道数、重复块的次数和应用于重复块的第一个序列的步幅参数。水平线分隔了模块：学习下采样、全局特征提取器、特征融合和分类器（从上到下）。</center>
</div>

## 3.1 动机

当前实时运行的最先进的语义分割方法基于具有两个分支的网络，每个分支在不同的分辨率级别上运行。它们从输入图像的低分辨率版本中学习全局信息，并使用完整输入分辨率的浅层网络来优化分割结果的精度。由于输入分辨率和网络深度是运行时的主要因素，这些两个分支的方法可以实现实时计算。众所周知，DCNN 的前几层提取低级特征，如边缘和角点。因此，我们引入了学习下采样的方法，它在浅层网络块中共享低级和高级分支之间的特征计算，而不是采用两个分支的独立计算方法。

## 3.2 网络架构

我们的 Fast-SCNN 使用了一个学习下采样模块、一个粗糙的全局特征提取器、一个特征融合模块和一个标准分类器。所**有模块都是使用深度可分离卷积构建的**，深度可分离卷积已经成为许多高效 DCNN 架构的关键构建模块。

### 3.2.1 Learning to Downsample，可学习的下采样模块

在我们的学习下采样模块中，我们使用了三个层次。只使用三个层次是为了确保低级特征共享的有效性和高效实现。第一层是标准卷积层(Conv2D)，剩下的两层是深度可分离卷积层(DSConv)。在这里，我们强调一下，虽然 DSConv 在计算上更高效，但我们使用 Conv2D 是因为输入图像只有三个通道，在这个阶段 DSConv 的计算优势微不足道。

我们学习下采样模块中的所有三个层次都使用步幅为 2，接着进行 BN 和 ReLU 操作。卷积和深度可分离层的空间核大小为 3×3。需要注意的是，我们在深度可分离卷积和逐点卷积之间省略了非线性操作。

### 3.2.2 Global Feature Extractor，全局特征提取器

全局特征提取器模块旨在捕捉图像分割的全局上下文。与通常在输入图像的低分辨率版本上操作的常见两个分支方法不同，我们的模块直接使用学习下采样模块的输出（位于原始输入的 1/8 分辨率）。模块的详细结构如[表 1](#table1) 所示。我们使用了 MobileNet-V2 引入的高效瓶颈残差块（[表 2](#table2)）。特别地，当输入和输出大小相同时，我们为瓶颈残差块使用了残差连接。我们的瓶颈块使用了高效的深度可分离卷积，从而减少了参数和浮点运算的数量。此外，我们在末尾添加了金字塔池化模块（PPM），以聚合不同区域的上下文信息。

<div align=center>
    <img src=./imgs_markdown/2024-03-15-09-47-54.png
    width=60%><a id='table2'></a>
    <center></center>
</div>

表格 2. 瓶颈残差块将输入从 $c$ 通道转换为 $c'$ 通道，扩展因子为 $t$。注意，最后的逐点卷积不使用非线性函数 $f$。输入的高度为 $h$，宽度为 $w$，x/$s$ 表示层的核大小和步幅。</br>

### 3.2.3 Feature Fusion Module, 特征融合模块

类似于 ICNet 和 ContextNet，我们更倾向于简单地将特征相加以确保效率。或者，可以使用更复杂的特征融合模块以达到更好的准确性，但会牺牲运行时性能。特征融合模块的详细信息如[表 3](#table3) 所示。

<div align=center>
    <img src=./imgs_markdown/2024-03-15-09-51-42.png
    width=60%><a id='table3'></a>
    <center></center>
</div>

表 3. Fast-SCNN 的特征融合模块（FFM）。注意，逐点卷积的输出是所需的，不使用非线性函数 $f$。在特征相加之后使用非线性函数 $f$。</br>

### 3.2.4 Classifier，分类器

在分类器中，我们使用了两个深度可分离卷积（DSConv）和一个逐点卷积（Conv2D）。我们发现，在特征融合模块之后添加几层可以提高准确性。分类器模块的详细信息如[表 1](#table1) 所示。

在训练过程中，我们使用 Softmax 函数，因为我们使用了梯度下降算法。在推断过程中，我们可以用 argmax 函数替代计算成本较高的 Softmax 函数，因为这两个函数都是单调递增的。我们将这个选项称为 Fast-SCNN cls（分类）。另一方面，如果需要基于标准 DCNN 的概率模型，则使用 Softmax 函数，称为 Fast-SCNN prob（概率）。

## 3.3 与先前的方法比较

我们的模型受到了两个分支框架的启发，并结合了编码器-解码器方法的思想（如[图 2](#fig2) 所示）。

### 3.3.1 与两个分支模型的关系

最先进的实时模型（ContextNet、BiSeNet和GUN）使用了两个分支网络。我们的学习下采样模块等效于它们的空间路径，因为它是浅层的，从全分辨率学习，并在特征融合模块中使用（如[图 1](#fig1)）。

我们的全局特征提取器模块等效于这些方法中更深的低分辨率分支。与此相反，我们的全局特征提取器与学习下采样模块共享前几层的计算。通过共享层，我们不仅减少了特征提取的计算复杂度，还减小了 Fast-SCNN 对输入尺寸的要求，因为它在全局特征提取时使用的是 1/8 分辨率而不是 1/4 分辨率。

### 3.3.2 与编码器-解码器模型的关系

提出的 Fast-SCNN 可以被视为编码器-解码器框架（如 FCN 或 U-Net）的特例。然而，与 FCN 中的多个跳跃连接和 U-Net 中的密集跳跃连接不同，Fast-SCNN 只使用一个跳跃连接来减少计算和内存消耗。

> 与我想的一样，只有一个分支

与 [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901) 的观点相一致，他们主张在 DCNN 中只在早期层次上共享特征，我们将我们的跳跃连接放在网络的早期位置。相比之下，先前的方法通常在每个分辨率上使用更深的模块，然后再应用跳跃连接。

# 4. 实验

我们在 Cityscapes 数据集的验证集上评估了 Fast-SCNN，并在 Cityscapes 测试集上报告了其性能，即 Cityscapes 基准服务器。

## 4.1 实现细节

实现细节与理论一样重要，尤其是在实现高效的 DCNN 时。因此，我们在这里仔细描述我们的设置。我们在 Python 中使用 TensorFlow 机器学习平台进行实验。我们的实验在一台工作站上执行，配备了 Nvidia Titan X（Maxwell）或 Nvidia Titan Xp（Pascal）GPU，使用 CUDA 9.0 和 CuDNN v7。运行时评估在单个 CPU 线程和一个 GPU 上执行，以测量前向推断时间。我们使用 100 帧进行热身，然后报告 100 帧的平均帧率（fps）测量结果。

我们使用带有动量 0.9 和批量大小为 12 的随机梯度下降（SGD）算法。我们使用“poly”学习率，基础学习率为 0.045，幂次为 0.9。与 MobileNet-V2 类似，我们发现深度可分离卷积不需要 $\ell_2$ 正则化，对于其他层，$\ell_2$ 正则化率为 0.00004。由于语义分割的训练数据有限，我们应用了各种数据增强技术：随机调整大小（0.5 到 2 之间）、平移/裁剪、水平翻转、颜色通道噪声和亮度调整。我们的模型使用交叉熵损失进行训练。我们发现，在学习下采样和全局特征提取模块的末尾使用 0.4 权重的辅助损失对模型有益。

Batch normalization 在每个非线性函数之前使用。Dropout 仅在最后一层的 softmax 层之前使用。与 MobileNet 和 ContextNet 相反，我们发现 Fast-SCNN 在使用 ReLU 激活函数时训练速度更快，并且在整个模型中使用的深度可分离卷积下，准确性略高于 ReLU6。

我们发现通过训练更多的迭代次数可以提高 DCNN 的性能，因此我们将模型训练了 1,000 个 epochs，除非另有说明，使用 Cityscapes 数据集。值得注意的是，Fast-SCNN 的容量故意设置得非常低，因为我们只使用了 111 万个参数。后面我们将展示，通过采用激进的数据增强技术，过拟合的可能性很低。

## 4.2 在 Cityscapes 数据集上的评估

我们在 Cityscapes 上评估了我们提出的 Fast-SCNN，这是最大的公开可用的城市道路数据。该数据集包含来自欧洲 50 个不同城市的多样化高分辨率图像（1024×2048 像素）。其中，有 5000 张具有高质量标签的图像，包括 2975 张训练集、500 张验证集和 1525 张测试集。训练集和验证集的标签是可用的，而测试结果可以在评估服务器上进行评估。此外，还有 2 万张弱标注图像（粗略标签）可用于训练。我们同时报告了使用细粒度标签和细粒度与粗略标签数据的结果。Cityscapes 提供了 30 个类别标签，但只有 19 个类别用于评估。我们报告了交并比均值（mIoU）和网络推断时间。

我们在 Cityscapes 的保留测试集上评估了 Fast-SCNN 的整体性能。在[表 4](#table4) 中，我们将 Fast-SCNN 与其他最先进的实时语义分割方法（ContextNet、BiSeNet、GUN 、ENet 和 ICNet）以及离线方法（PSPNet 和 DeepLab-V2）进行了比较。Fast-SCNN 实现了 68.0%的 mIoU，略低于 BiSeNet（71.5%）和 GUN（70.4%）。ContextNet 在这里只达到了 66.1%。

<div align=center>
    <img src=./imgs_markdown/2024-03-15-10-11-14.png
    width=60%><a id='table4'></a>
    <center>表 4. 在 Cityscapes 测试集上，与其他最先进的语义分割方法相比，所提出的 Fast-SCNN 的类别 mIoU 和总 mIoU。参数数量以百万为单位列出。</center>
</div></br>

<div align=center>
    <img src=./imgs_markdown/2024-03-15-10-10-31.png
    width=70%><a id='table5'></a>
    <center>表 5. 在 Nvidia Titan X（Maxwell，3072 个 CUDA 核心）上使用 TensorFlow 的运行时间（fps）。带有 * 的方法表示在 Nvidia Titan Xp（Pascal，3840 个 CUDA 核心）上的结果。显示了 Fast-SCNN 的两个版本：使用 softmax 输出（我们的 prob）和使用对象标签输出（我们的 cls）。</center>
</div></br>

[表 5](#table5) 比较了不同分辨率下的运行时间。在这里，BiSeNet（57.3 fps）和 GUN（33.3 fps）比 Fast-SCNN（123.5 fps）慢得多。与 ContextNet（41.9 fps）相比，Fast-SCNN 在 Nvidia Titan X（Maxwell）上也显著更快。因此，我们得出结论，Fast-SCNN 在轻微准确性损失的情况下显著改进了最先进的运行时间。值得强调的是，我们的模型是为低内存嵌入式设备设计的。Fast-SCNN 使用了 111 万个参数，比竞争对手 BiSeNet 的 580 万个参数少了五倍。

最后，我们将跳跃连接的贡献置零，并测量 Fast-SCNN 的性能。在验证集上，mIoU 从 69.22% 降至 64.30%。在[图 3](#fig3) 中，我们比较了定性结果。正如预期的那样，Fast-SCNN 在边界和小尺寸物体周围，特别受益于跳跃连接的作用。

<div align=center>
    <img src=./imgs_markdown/2024-03-15-10-28-09.png
    width=100%><a id='fig3'></a>
    <center>图 3. Fast-SCNN 的分割结果可视化。第一列：输入的 RGB 图像；第二列：Fast-SCNN 的输出；最后一列：在将跳跃连接的贡献置零后的 Fast-SCNN 的输出。在所有的结果中，Fast-SCNN 特别在边界和小尺寸物体上受益于跳跃连接。</center>
</div></br>

## 4.3 预训练和弱标注数据

高容量的 DCNN，如 R-CNN 和 PSPNet，已经证明通过不同的辅助任务可以提升性能。由于我们专门设计了低容量的 Fast-SCNN，现在我们想测试在是否进行预训练以及是否使用额外的弱标注数据的情况下的性能。据我们所知，以往没有研究过预训练和额外弱标注数据对低容量 DCNN 的影响。[表 6](#table6) 显示了结果。

<div align=center>
    <img src=./imgs_markdown/2024-03-15-10-20-13.png
    width=60%><a id='table6'></a>
    <center>表 6. Cityscapes 验证集上不同 Fast-SCNN 设置的类别 mIoU</center>
</div></br>

我们使用 ImageNet 对 Fast-SCNN 进行预训练，通过将特征融合模块替换为平均池化，并且分类模块只有一个 softmax 层。Fast-SCNN 在 ImageNet 验证集上达到了 60.71% 的 top-1 准确率和 83.0% 的 top-5 准确率。这个结果表明，Fast-SCNN 的容量不足以达到 ImageNet 上大多数标准 DCNN 的性能水平（> 70% 的 top-1 准确率）。使用 ImageNet 预训练的 Fast-SCNN 在 Cityscapes 验证集上的 mIoU 为 69.15%，仅比没有预训练的 Fast-SCNN 提高了 0.53%。因此，我们得出结论，<font color='red'><b>Fast-SCNN 在 ImageNet 预训练方面无法获得显著的提升</b></font>。

由于 Cityscapes 的城市道路和 ImageNet 的分类任务之间的重叠有限，可以合理地假设 Fast-SCNN 可能由于两个领域的容量有限而无法受益。因此，我们现在加入了 Cityscapes 提供的 2 万个粗略标注的额外图像，因为它们来自类似的领域。然而，使用粗糙训练数据（带有或不带有 ImageNet）训练的 Fast-SCNN 之间表现相似，并且只略微改进了原始的没有预训练的 Fast-SCNN。请注意，小的变化是不显著的，这是由于 DCNN 的随机初始化造成的。

值得注意的是，使用辅助任务是非常复杂的，因为它需要在网络中进行架构修改。此外，许可限制和资源不足进一步限制了这样的设置。由于我们证明了 ImageNet 预训练和弱标注数据对于我们的低容量 DCNN 并没有显著的益处，因此可以节省这些成本。[图 4](#fig4) 显示了训练曲线。使用粗略数据的 Fast-SCNN 在迭代次数方面训练较慢，这是由于弱标签质量的影响。两个 ImageNet 预训练版本在早期时期表现较好（仅使用训练集训练时达到 400 个时期，使用额外的粗略标注数据训练时达到 100 个时期）。这意味着，当我们从头开始训练模型时，我们只需要更长时间的训练才能达到类似的准确性。

<div align=center>
    <img src=./imgs_markdown/2024-03-15-10-22-31.png
    width=80%><a id='fig4'></a>
    <center>图 4. Cityscapes 上的训练曲线。上方显示了迭代次数 (Iterations) 的准确率，下方显示了 Epoch 的准确率。虚线表示 Fast-SCNN 的 ImageNet 预训练。</center>
</div></br>

## 4.4 降低输入分辨率

由于我们对可能没有完整分辨率输入或无法访问强大 GPU 的嵌入式设备感兴趣，我们通过在一半和四分之一的输入分辨率下进行性能研究来结束我们的评估（见[表 7](#table7)）。

<div align=center>
    <img src=./imgs_markdown/2024-03-15-10-24-56.png
    width=45%><a id='table7'></a>
    <center>表7. Fast-SCNN在Cityscapes测试集上不同输入分辨率下的运行时间和准确率</center>
</div></br>

在四分之一的分辨率下，Fast-SCNN 以 485.4 fps 的速度达到了 51.9% 的准确率，这显著优于（匿名）MiniNet 在 250 fps 下的 40.7% 的 mIoU。在一半的分辨率下，达到了竞争力强的 285.8 fps 下的 62.8% 的 mIoU。

我们强调，<font color='red'><b>Fast-SCNN 无需修改即可直接适用于较低的输入分辨率，使其非常适用于嵌入式设备</b></font>。

<div align=center>
    <img src=./imgs_markdown/2024-03-15-10-30-57.png
    width=100%><a id='fig5'></a>
    <center>图 5. Fast-SCNN 在 Cityscapes 验证集上的定性结果。第一列：输入的 RGB 图像；第二列：真实标签；最后一列：Fast-SCNN 的输出。Fast-SCNN 获得了 68.0% 的类别级别 mIoU 和 84.7% 的类别级别 mIoU。</center>
</div></br>

# 5. 结论

我们提出了一种用于实时场景理解的快速分割网络。通过共享多分支网络的计算成本，实现了运行时的效率。在实验中，我们证明了跳跃连接对于恢复空间细节是有益的。我们还证明了，对于低容量网络来说，如果训练时间足够长，不需要在额外的辅助任务上进行大规模预训练模型。

# 6. 模型代码

这里使用的是 [他人复现的 PyTorch 代码](https://github.com/Tramac/Fast-SCNN-pytorch)，并非 PyTorch 官方复现的代码。

下面是模型定义：

```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FastSCNN', 'get_fast_scnn']


class FastSCNN(nn.Module):
    def __init__(self, num_classes, aux=False):
        super(FastSCNN, self).__init__()
        self.aux = aux  # 是否使用辅助分类头
        
        # 定义可学习的下采样模块(其实就是会进行下采样的卷积)
        self.learning_to_downsample = LearningToDownsample(
            dw_channels1=32,
            dw_channels2=48,
            out_channels=64)

        # 定义特征提取模块
        self.global_feature_extractor = GlobalFeatureExtractor(
            in_channels=64,
            block_channels=[64, 96, 128],
            out_channels=128,
            t=6,
            num_blocks=[3, 3, 3])
        
        # 定义特征融合模块
        self.feature_fusion = FeatureFusionModule(
            highter_in_channels=64, 
            lower_in_channels=128, 
            out_channels=128)
        
        self.classifier = Classifer(128, num_classes)
        
        # 定义辅助分类头
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes, 1)
            )

    def forward(self, x):
        size = x.size()[2:]
        
        # 先让特征图经过可学习的下采样模块 -> 作为一个分支 --> X1
        higher_res_features = self.learning_to_downsample(x)

        # 让特征图正常进行 --> X2
        x = self.global_feature_extractor(higher_res_features)
        
        # 将 X1 和 X2 融合
        x = self.feature_fusion(higher_res_features, x)
        
        # 分类头
        x = self.classifier(x)

        # 上采样恢复到和原图一样的大小
        outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        
        # 如果使用了辅助分类头，那么结果有两个
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(
                auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
            
        return tuple(outputs)


class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride,
                      1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride,
                      1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]  # 获取 H, W
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(
            3, dw_channels1, kernel_size=3, stride=2, padding=0)  # 步长为2
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, stride=2)  # 步长为2
        self.dsconv2 = _DSConv(dw_channels2, out_channels, stride=2)  # 步长为2

    def forward(self, x):
        x = self.conv(x)  # 两倍下采样
        x = self.dsconv1(x)  # 两倍下采样
        x = self.dsconv2(x)  # 两倍下采样

        # 此时特征图经过了 6倍 下采样
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3), **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        
        self.bottleneck1 = self._make_layer(
            LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(
            LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(
            LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(
            lower_res_feature, scale_factor=4, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x


def get_fast_scnn(dataset='citys', pretrained=False, root='./weights', map_cpu=False, **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from data_loader import datasets
    model = FastSCNN(datasets[dataset].NUM_CLASS, **kwargs)
    if pretrained:
        if (map_cpu):
            model.load_state_dict(torch.load(os.path.join(
                root, 'fast_scnn_%s.pth' % acronyms[dataset]), map_location='cpu'))
        else:
            model.load_state_dict(torch.load(os.path.join(
                root, 'fast_scnn_%s.pth' % acronyms[dataset])))
    return model


if __name__ == '__main__':
    img = torch.randn(2, 3, 256, 512)
    model = get_fast_scnn('citys')
    outputs = model(img)
```

上面的代码也没啥可说的。