# 5. YOLOv5 训练技巧

## 5.1 warm-up

在 YOLOv5 中，warm-up（预热）是指在训练初始阶段使用较小的学习率，然后逐渐增加学习率，以帮助模型更好地适应数据集。这个过程有助于避免在初始阶段出现梯度爆炸或不稳定的情况，使模型更容易收敛。

YOLOv5 中的 warm-up 主要体现在学习率的调整上。具体而言，YOLOv5 使用线性 warm-up 策略，即在初始训练阶段，学习率从一个较小的初始值线性增加到设定的初始学习率。这有助于减缓模型的参数更新速度，防止在初始时出现过大的权重更新，从而提高训练的稳定性。

在 YOLOv5 的实现中，warm-up 阶段通常持续一定的迭代次数，这个次数是在训练开始时设定的。一旦 warm-up 阶段结束，模型将以设定的初始学习率进行正常的训练。

Warm-up 的主要优势在于可以在模型开始学习任务时更好地控制学习的速度，从而有助于模型更快地适应数据分布。这在处理复杂的目标检测任务中尤为重要，因为这些任务通常具有大量的样本和复杂的背景。

我们看一下相关的源码（`train.py`）：

```python
nb = len(train_loader)  # number of batches | 一个epoch拥有的batch数量
nw = max(round(hyp["warmup_epochs"] * nb), 100)  # number of warmup | 热身的总迭代次数

pbar = enumerate(train_loader)  # 遍历train_loader

# 记录日志
LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "Instances", "Size"))

# 如果在主线程中，那么给enumberate加上tqdm进度条
if RANK in {-1, 0}:
    pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar

# 开始遍历train_loader
for i, (imgs, targets, paths, _) in pbar:  # batch 
    # imgs: 一个batch的图片
    # targets: 一个batch的标签
    # paths: 一个batch的路径
    callbacks.run("on_train_batch_start")  # 记录此时正在干什么

    # 计算当前的迭代次数
    ni = i + nb * epoch  # number integrated batches (since train start)
    imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

    # Warmup
    if ni <= nw:  # 如果当前的迭代次数小于需要热身的迭代次数，则开始热身
        xi = [0, nw]  # x interp

        # accumulate变量的作用是动态地控制累积的 Batch 数，以便在训练开始时逐渐增加累积的 Batch 数，
        # 从而实现从较小的累积 Batch 数到较大的累积 Batch 数的平滑过渡
        # 这有助于模型在训练初期稳定地学习
        accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
        for j, x in enumerate(optimizer.param_groups):
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
            if "momentum" in x:
                x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])
```

在 [How suspend image mixing #931](https://github.com/ultralytics/yolov5/issues/931) 中有作者关于 warm-up 的说明：

warmup 慢慢地将训练参数从它们的初始（更稳定）值调整到它们的默认训练值。例如，通常会在最初的几个 Epoch 内将学习率从 0 调整到某个初始值，以避免早期训练的不稳定、nan 等问题。

热身效果可以在 Tensorboard 的学习率曲线图中观察到，这些曲线自从最近的提交以来已经被自动跟踪。下面的例子显示了在自定义数据集上大约 30 个 Epoch 的热身，每个参数组有一个曲线图。最后一个曲线图展示了不同的热身策略（即不同的超参数设置）。

<div align=center>
    <img src=./imgs_markdown/2024-02-06-17-35-38.png
    width=100%>
    <center></center>
</div>

### 5.1.1 np.interp 语法

`numpy.interp(x, xp, fp, left=None, right=None, period=None)` 是 NumPy 中的一个函数，用于线性插值。线性插值是一种估算在两个已知值之间的未知值的方法，假设这些值之间的变化是线性的。

其中：

- `x`: 需要进行插值的一维数组。
- `xp`: 已知数据点的 x 坐标（一维数组）-> x points。
- `fp`: 已知数据点的 y 坐标（一维数组）-> function points。
- `left`: 当 x 小于 xp 的最小值时，返回的默认值，默认为 fp[0]。
- `right`: 当 x 大于 xp 的最大值时，返回的默认值，默认为 fp[-1]。
- `period`: 如果提供了 period，表示 xp 是周期性的，此时插值会考虑周期性。period 是周期的长度。

**示例**：

```python
import numpy as np
import matplotlib.pyplot as plt

# 已知数据点
x_known = np.array([1, 2, 3, 4, 5])
y_known = np.array([3, 5, 7, 9, 11])

# 待插值的数据点
x_unknown = [0.0, 1.5, 3.0, 4.5, 6.0]

# 使用np.interp进行插值
y_unknown = np.interp(x_unknown, x_known, y_known)
print(f"{y_unknown = }")  # [3, 4, 7, 10, 11]

# 绘制图形
plt.figure(figsize=(10, 6), dpi=200)
plt.plot(x_known, y_known, 'o', label='Known points', color='green')  # 已知数据点
plt.plot(x_unknown, y_unknown, 'o', label='Unknown points', color='red')  # 插值结果
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'Example for $np.interp()$')
plt.legend()
plt.grid(True)
plt.savefig('Example4np.interp.jpg')
```

看不懂没关系，我们作图看一下：

<div align=center>
    <img src=./imgs_markdown/2024-02-05-14-29-23.png
    width=100%>
    <center></center>
</div>

外推规则如下：
- 如果 `x` 的值小于 `xp` 的最小值，则 `np.interp` 返回与 `xp` 最小值对应的 `fp` 值。
- 如果 `x` 的值大于 `xp` 的最大值，则 `np.interp` 返回与 `xp` 最大值对应的 `fp` 值。

分析如下：

- x[0] = 0.0, 它小于 xp 的最小值1，所以外推，此时 x[0] 对应的 y[0] = fp[0] -> 3
- x[1] = 1.5, 它在 xp 的 [1, 2] 之间，所以对应的 y[1] 应该为 y[0] = (fp[1] + f[2]) / 2 --> (3 + 5) / 2 = 4
- x[2] = 3.0 == xp[2], 所以对应的 y[2] == fp[2] --> 7
- x[3] = 4.5 ∈ [4, 5], y[3] == (fp[4] + fp[5]) / 2 --> (9 + 11) / 2 --> 10
- x[4] = 6.0，它大于 xp 的最大值，所以外推，此致 x[4] 对应的 y[4] == fp[5] --> 11

> ⚠️ x 和 y 取的是索引，而 xp 和 fp 这里不是取索引，而是取值

## 5.2 Cosine Annealing Warm Restart

- 论文地址：[SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
- 翻译：[Cosine Annealing Warm Restart论文讲解](https://blog.csdn.net/weixin_44878336/article/details/125016166)

Cosine Annealing Warm Restart 是一种学习率调度策略，它是基于余弦退火周期性调整学习率的算法。这种策略在学习率调整上引入了周期性的“重启”，使得模型在训练过程中能够周期性地跳出局部最小值，从而有助于提高模型的泛化能力和性能。

具体来说，Cosine Annealing Warm Restart 策略包括以下几个关键组成部分：

1. **余弦退火周期**：在每个周期内，学习率按照余弦函数的变化规律进行调整。余弦函数从最大值开始，逐渐减小到最小值，因此学习率也会从初始值开始，先减小到一个低点，然后再增加回到初始值。
2. **周期性重启**：在每个周期结束时，学习率会被重新设置回初始值，并重新开始一个新的周期。这种重启有助于模型跳出当前的优化路径，探索新的参数空间。
3. **周期长度调整**：随着训练的进行，周期长度（即退火周期）和最小学习率可以逐渐调整。通常，每个周期的长度会逐渐减小，而最小学习率会逐渐增加，这样可以让模型在训练后期更加细致地搜索最优解。
4. **学习率范围**：在每个周期内，学习率的变化范围是从最大值到最小值，这两个值都可以根据实际情况进行调整。

Cosine Annealing Warm Restart 策略的优势在于它通过周期性重启和调整周期长度，使得模型能够在训练过程中不断探索新的参数空间，从而有可能找到更好的局部最小值或全局最小值。这种策略特别适合于那些容易陷入局部最小值的复杂模型训练，可以提高模型的最终性能和泛化能力。

在论文 [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187) 中有介绍到余弦退火和阶段两种学习率在 ImageNet 数据集上的表现（模型为 ResNet-50）：

<div align=center>
    <img src=./imgs_markdown/2024-02-06-17-42-37.png
    width=70%>
    <center></center>
</div>

> 图 3：带有热身阶段的学习率计划的可视化。顶部：Batch size=1024 下的余弦和阶跃调度。底部：两种调度下的Top-1验证准确率曲线。

---

余弦退火热重启的调用如下：

```python
import torch.optim as optim


model = ...
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 使用CosineAnnealingWarmRestarts调度器
# T_0是初始周期的大小，T_mult每个周期结束后周期大小乘以的倍数
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

for epoch in range(num_epochs):
    # 训练模型的代码
    train(...)
    # 在每个epoch后更新学习率
    scheduler.step(epoch)
```

在上面的代码中，`T_0` 参数代表初始周期的大小，即在第一次余弦退火周期中，学习率将按照余弦调度进行调整的 Epoch 数。`T_mult` 参数指定了每个周期结束后周期大小将乘以的倍数。`scheduler.step(epoch)` 应该在每次更新参数之后、每个epoch结束时调用。
请根据你的具体需求调整 `T_0` 和 `T_mult` 的值，以及 `num_epochs`，即你的训练周期总数。

## 5.3 YOLOv5-v7.0 使用的 Scheduler

```python
# Scheduler
if opt.cos_lr:  # 如果使用cosine学习率
    lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
else:
    lf = lambda x: (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
```

我们画图看一下二者的区别：

```python
import matplotlib.pyplot as plt
import math


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


# 设定训练的总epoch数
epochs = 100

# YOLOv5中的超参数
hyp = {
    "lr0": 0.01,  # 初始学习率
    "lrf": 0.1  # final OneCycleLR learning rate (lr0 * lrf)
}

# 创建一个numpy数组，表示epoch数
epoch_lst = range(epochs)

# Cosine调度器的学习率变化
lf_cos = one_cycle(1, hyp["lrf"], epochs)
lr_cos = [lf_cos(epoch) for epoch in epoch_lst]

# Linear调度器的学习率变化
lf_lin = lambda x: (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]
lr_lin = [lf_lin(epoch) for epoch in epoch_lst]

# 绘制学习率变化曲线
plt.figure(figsize=(10, 6), dpi=200)

plt.plot(epoch_lst, lr_cos, '-', label='Cosine Scheduler', color='skyblue')
plt.plot(epoch_lst, lr_lin, '-.', label='Linear Scheduler', color='lightpink')

plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Comparison of Cosine and Linear Learning Rate Schedulers')

plt.legend()
plt.grid(True)
plt.savefig('Le0v1n/results/Comparison-of-Cosine-and-Linear-Learning-Rate-Schedulers.jpg')
```

<div align=center>
    <img src=./imgs_markdown/Comparison-of-Cosine-and-Linear-Learning-Rate-Schedulers.jpg
    width=100%>
    <center></center>
</div>

## 5.3 AutoAnchor

### 5.3.1 目的

AutoAnchor 是 YOLOv5 中的一个功能，用于自动调整 Anchor（anchor boxes）的大小以更好地适应训练数据集中的对象形状。

> Anchor 是在对象检测任务中使用的一种技术，它们代表了不同大小和宽高比的预定义边界框，用于预测真实对象的位置和大小。

在 YOLOv5 中，AutoAnchor 的主要目的是优化 Anchor 的大小，以便在训练期间提高检测精度和效率。这个功能在训练过程开始时执行，根据训练数据集中的边界框计算最佳 Anchor 配置。通过这种方式，YOLOv5 可以自动适应新的数据集，而无需手动调整 Anchor。

### 5.3.2 AutoAnchor 的步骤

1. **分析数据集**：分析数据集中的边界框，了解对象的大小和形状分布。
2. **Anchor聚类**：使用聚类算法（如 K-means）对边界框进行聚类，以确定最佳的 Anchor 数量和大小。
3. **更新配置**：根据聚类结果更新 Anchor 配置，以便在训练期间使用这些新 Anchor。
4. **重新训练**：使用新的 Anchor 配置重新开始训练过程。

### 5.3.3 作用

AutoAnchor 的优势在于它能够为特定的数据集定制 Anchor，这有助于提高检测精度，尤其是在处理具有不同对象大小和形状的多样化数据集时。通过自动调整 Anchor，YOLOv5 可以更有效地利用计算资源，减少对超参数的手动调整需求，从而简化了模型训练过程。

### 5.3.4 源码

首先需要先计算当前的 Anchor 与数据集的适应程度。

```python
@TryExcept(f"{PREFIX}ERROR")
def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # 函数作用：检查anchor是否适合数据，如有必要，则重新计算anchor

    # 从模型中获取检测层（Detect()）
    m = model.module.model[-1] if hasattr(model, "module") else model.model[-1]
    
    # 计算输入图片的尺寸相对于最大尺寸的比例
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)

    # 生成一个随机的比例因子，用于扩大或缩小图片尺寸
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))

    # 计算所有图片的宽高（wh）
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)]))

    def metric(k):  # 计算度量值
        # 计算每个anchor与gt boxes的宽高比
        r = wh[:, None] / k[None]

        # 计算最小比率和最大比率
        x = torch.min(r, 1 / r).min(2)[0]

        # 找到最大比率的anchor
        best = x.max(1)[0]

        # 计算超过阈值（thr）的anchor数量占比
        aat = (x > 1 / thr).float().sum(1).mean()

        # 计算BPR（best possible recall）
        bpr = (best > 1 / thr).float().mean()

        return bpr, aat

    # 获取模型的步长（stride）
    stride = m.stride.to(m.anchors.device).view(-1, 1, 1)

    # 计算当前的anchor
    anchors = m.anchors.clone() * stride
    
    # 计算当前anchor与gt boxes的比值，并找到最佳比值和超过阈值的anchor占比
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    s = f"\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). "
    
    # 如果最佳比值召回率大于0.98，说明当前anchor适合数据集
    if bpr > 0.98:
        LOGGER.info(f"{s}Current anchors are a good fit to dataset ✅")
    else:  # 说明anchor不适合数据集，需要尝试改进
        LOGGER.info(f"{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...")

        # 计算anchor数量
        na = m.anchors.numel() // 2

        # 使用k-means聚类算法重新计算anchor
        anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)

        # 计算新anchor的最佳比值召回率
        new_bpr = metric(anchors)[0]

        # 如果新anchor的召回率比原来的高，则替换anchor
        if new_bpr > bpr:
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors)

            # 检查anchor顺序是否正确（必须在像素空间，不能在网格空间）
            check_anchor_order(m)
            m.anchors /= stride
            s = f"{PREFIX}Done
```

## 5.4 Hyper-parameter Evolution 超参数进化

超参数进化（Hyperparameter Evolution）是一种模型优化技术，它涉及在训练过程中动态地调整模型的超参数（hyperparameters），以找到在特定数据集上性能最佳的参数设置。这些超参数是模型设计中的高级设置，它们控制模型的学习过程，但不直接作为模型输入的一部分。常见的超参数包括学习率、批量大小、迭代次数、正则化参数、Anchor 大小等。

超参数进化的目标是减少超参数调整的试错过程，提高模型训练的效率。传统的超参数调整方法通常需要手动调整超参数或使用网格搜索（Grid Search）等方法进行大量的实验来找到最佳设置。这些方法既耗时又可能无法找到最优解。

在 [《超参数演变》](https://docs.ultralytics.com/zh/yolov5/tutorials/hyperparameter_evolution/) 这一官方文档中对其进行了介绍：

超参数演化是一种使用遗传算法（GA）进行优化的超参数优化方法。

ML 中的超参数控制着训练的各个方面，而为超参数寻找最佳值是一项挑战。网格搜索等传统方法很快就会变得难以处理，原因在于：1）搜索空间维度高；2）维度之间的相关性未知；3）评估每个点的适配性成本高昂，因此 GA 是超参数搜索的合适候选方法。

GA 的流程如下：

<div align=center>
    <img src=./imgs_markdown/plots-GA.jpg
    width=80%>
    <center></center>
</div>

我们看一下官方的介绍：

### 5.4.1 初始化超参数

YOLOv5 有大约 30 个超参数，用于不同的训练设置。这些参数在 `*.yaml` 文件中的 `/data/hyps` 目录。更好的初始猜测将产生更好的最终结果，因此在演化之前正确初始化这些值非常重要。如果有疑问，只需使用默认值即可，这些值已针对 YOLOv5 COCO 从头开始的训练进行了优化。

```yaml
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
# Hyperparameters for low-augmentation COCO training from scratch

lr0: 0.01 # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.01 # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937 # SGD momentum/Adam beta1
weight_decay: 0.0005 # optimizer weight decay 5e-4
warmup_epochs: 3.0 # warmup epochs (fractions ok)
warmup_momentum: 0.8 # warmup initial momentum
warmup_bias_lr: 0.1 # warmup initial bias lr
box: 0.05 # box loss gain
cls: 0.5 # cls loss gain
cls_pw: 1.0 # cls BCELoss positive_weight
obj: 1.0 # obj loss gain (scale with pixels)
obj_pw: 1.0 # obj BCELoss positive_weight
iou_t: 0.20 # IoU training threshold
anchor_t: 4.0 # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0 # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015 # image HSV-Hue augmentation (fraction)
hsv_s: 0.7 # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4 # image HSV-Value augmentation (fraction)
degrees: 0.0 # image rotation (+/- deg)
translate: 0.1 # image translation (+/- fraction)
scale: 0.5 # image scale (+/- gain)
shear: 0.0 # image shear (+/- deg)
perspective: 0.0 # image perspective (+/- fraction), range 0-0.001
flipud: 0.0 # image flip up-down (probability)
fliplr: 0.5 # image flip left-right (probability)
mosaic: 1.0 # image mosaic (probability)
mixup: 0.0 # image mixup (probability)
copy_paste: 0.0 # segment copy-paste (probability)
```

### 5.4.2 定义适应度（fitness）

适应度是我们试图最大化的值。在 YOLOv5 中，我们定义了一个默认的适应度函数，它是以下指标的加权组合：`mAP@0.5` 贡献了 10% 的权重，而 `mAP@0.5:0.95` 贡献了剩余的 90%，其中不包括 Precision `P` 和 Recall `R`。我们可以根据需要调整这些指标，或者使用 `utils/metrics.py` 中的默认适应度定义（建议使用）。

```python
def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)
```

简单来说：

$$
\mathrm{fitness} = 0.1 \times \mathrm{mAP^{0.5}} + 0.9 \times \mathrm{mAP^{0.5:0.95}}
$$

### 5.4.3 进化（Evolve）

进化是基于我们寻求改进的基础情景进行的。在这个例子中，基础情景是在 COCO128 上对预训练的 YOLOv5s 进行 10 个周期的微调。基础情景的训练命令是：

```bash
python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache
```

为了针对这个情景进化特定的超参数，从我们在 5.4.1 中定义的初始值开始，并最大化我们在 5.4.2 中定义的适应度，请在命令行中添加 `--evolve` 参数：

```bash
# Single-GPU
python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --evolve

# Multi-GPU
for i in 0 1 2 3 4 5 6 7; do
  sleep $(expr 30 \* $i) &&  # 30-second delay (optional)
  echo 'Starting GPU '$i'...' &&
  nohup python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --device $i --evolve > evolve_gpu_$i.log &
done

# Multi-GPU bash-while (not recommended)
for i in 0 1 2 3 4 5 6 7; do
  sleep $(expr 30 \* $i) &&  # 30-second delay (optional)
  echo 'Starting GPU '$i'...' &&
  "$(while true; do nohup python train.py... --device $i --evolve 1 > evolve_gpu_$i.log; done)" &
done
```

> 💡 nohup 命令：`nohup` 是一个在 Unix-like 系统中常用的命令，用于在用户退出登录会话后继续运行命令。这个名字是 "no hang up" 的缩写，意味着即使会话挂起（即用户退出登录），命令也会继续执行。

默认的超参数进化设置将运行基础情景 300 次，即进行 300 代进化。我们可以通过 `--evolve` 参数修改代数，例如 `python train. py --evolve 1000`。

主要的遗传运算符是交叉（crossover）和变异（mutation）。在本研究中，变异被使用，变异概率为 80%，方差为 0.04，基于所有之前代中最佳父母组合创建新的后代。结果被记录到 `runs/evolve/exp/evolve.csv`，并且每一代中适应度最高的后代都被保存为 `runs/evolve/hyp_evolved.yaml`：

```yaml
# YOLOv5 Hyperparameter Evolution Results
# Best generation: 287
# Last generation: 300
#    metrics/precision,       metrics/recall,      metrics/mAP_0.5, metrics/mAP_0.5:0.95,         val/box_loss,         val/obj_loss,         val/cls_loss
#              0.54634,              0.55625,              0.58201,              0.33665,             0.056451,             0.042892,             0.013441

lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.2  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.5  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 1.0  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.1  # image translation (+/- fraction)
scale: 0.5  # image scale (+/- gain)
shear: 0.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.0  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 1.0  # image mosaic (probability)
mixup: 0.0  # image mixup (probability)
copy_paste: 0.0  # segment copy-paste (probability)
```

我们建议至少进行 300 代的进化以获得最佳效果。⚠️ 请注意，进化通常既昂贵又耗时，因为基础情景需要训练数百次，可能需要数百或数千小时的 GPU 时间。

### 5.4.4 可视化（Visualize）

`evolve.csv` 在进化完成之后，由 `utils.plots.plot_evolve()` 绘制为 `evolve.png`，每个超参数都有一个子图，显示适应度（y轴）与超参数值（x轴）的关系。黄色表示更高的浓度。垂直分布表明一个参数已被禁用且不会变异。这可以在 `train.py` 中的元字典中选择，对于固定参数并防止它们进化的场景非常有用。

<div align=center>
    <img src=./imgs_markdown/2024-02-07-14-31-43.png
    width=100%>
    <center></center>
</div>

### 5.4.5 源码

```python
# Hyperparameter evolution metadata (including this hyperparameter True-False, lower_limit, upper_limit)
# 超参数进化metadata（包括此超参数是否参与进化，下限，上限）
meta = {
    "lr0": (False, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
    "lrf": (False, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
    "momentum": (False, 0.6, 0.98),  # SGD momentum/Adam beta1
    "weight_decay": (False, 0.0, 0.001),  # optimizer weight decay
    "warmup_epochs": (False, 0.0, 5.0),  # warmup epochs (fractions ok)
    "warmup_momentum": (False, 0.0, 0.95),  # warmup initial momentum
    "warmup_bias_lr": (False, 0.0, 0.2),  # warmup initial bias lr
    "box": (False, 0.02, 0.2),  # box loss gain
    "cls": (False, 0.2, 4.0),  # cls loss gain
    "cls_pw": (False, 0.5, 2.0),  # cls BCELoss positive_weight
    "obj": (False, 0.2, 4.0),  # obj loss gain (scale with pixels)
    "obj_pw": (False, 0.5, 2.0),  # obj BCELoss positive_weight
    "iou_t": (False, 0.1, 0.7),  # IoU training threshold
    "anchor_t": (False, 2.0, 8.0),  # anchor-multiple threshold
    "anchors": (False, 2.0, 10.0),  # anchors per output grid (0 to ignore)
    "fl_gamma": (False, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
    "hsv_h": (True, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
    "hsv_s": (True, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
    "hsv_v": (True, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
    "degrees": (True, 0.0, 45.0),  # image rotation (+/- deg)
    "translate": (True, 0.0, 0.9),  # image translation (+/- fraction)
    "scale": (True, 0.0, 0.9),  # image scale (+/- gain)
    "shear": (True, 0.0, 10.0),  # image shear (+/- deg)
    "perspective": (True, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
    "flipud": (True, 0.0, 1.0),  # image flip up-down (probability)
    "fliplr": (True, 0.0, 1.0),  # image flip left-right (probability)
    "mosaic": (True, 0.0, 1.0),  # image mixup (probability)
    "mixup": (True, 0.0, 1.0),  # image mixup (probability)
    "copy_paste": (True, 0.0, 1.0),
}  # segment copy-paste (probability)

# GA configs
# 遗传算法的配置
pop_size = 50  # # 种群大小

# 变异率的最小值和最大值
mutation_rate_min = 0.01
mutation_rate_max = 0.5

# 交叉率的最小值和最大值
crossover_rate_min = 0.5
crossover_rate_max = 1

# 精英大小（保留的最好个体数量）的最小值和最大值
min_elite_size = 2
max_elite_size = 5

# 锦标赛大小（用于选择父代的选择池大小）的最小值和最大值
tournament_size_min = 2
tournament_size_max = 10

with open(opt.hyp, errors="ignore") as f:
    hyp = yaml.safe_load(f)  # load hyps dict

    # 如果在.yaml文件中没有 anchors 这个超参数，那么我们加上
    if "anchors" not in hyp:  # anchors commented in hyp.yaml
        hyp["anchors"] = 3

# 不使用AutoAnchors
if opt.noautoanchor:
    del hyp["anchors"], meta["anchors"]  # 从GA种群中删去

# 修改部分参数值
opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch

# 拼接保存路径
evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv"

# Delete the items in meta dictionary whose first value is False
# 删除元字典中其第一个值为 False 的项 --> 不参与进化的参数都删掉
del_ = [item for item, value_ in meta.items() if value_[0] is False]

# 在删除之前备份一下
hyp_GA = hyp.copy()  # Make a copy of hyp dictionary

# 开始删除不参与进化的超参数
for item in del_:
    del meta[item]  # Remove the item from meta dictionary
    del hyp_GA[item]  # Remove the item from hyp_GA dictionary

# Set lower_limit and upper_limit arrays to hold the search space boundaries
# 设置 lower_limit 和 upper_limit 数组以保持搜索空间的边界
lower_limit = np.array([meta[k][1] for k in hyp_GA.keys()])
upper_limit = np.array([meta[k][2] for k in hyp_GA.keys()])

# Create gene_ranges list to hold the range of values for each gene in the population
# 创建 gene_ranges 列表以持有种群中每个基因值的范围
gene_ranges = [(lower_limit[i], upper_limit[i]) for i in range(len(upper_limit))]

# Initialize the population with initial_values or random values
# 初始化种群，使用初始值或随机值
initial_values = []

# If resuming evolution from a previous checkpoint
# 根据之前的 ckpt 继续进化
if opt.resume_evolve is not None:
    assert os.path.isfile(ROOT / opt.resume_evolve), "evolve population path is wrong!"
    with open(ROOT / opt.resume_evolve, errors="ignore") as f:
        evolve_population = yaml.safe_load(f)
        for value in evolve_population.values():
            value = np.array([value[k] for k in hyp_GA.keys()])
            initial_values.append(list(value))

# If not resuming from a previous checkpoint, generate initial values from .yaml files in opt.evolve_population
# 如果不是从之前的ckpt恢复，则从 opt.evolve_population 中的 .yaml 文件生成初始值
else:
    yaml_files = [f for f in os.listdir(opt.evolve_population) if f.endswith(".yaml")]
    for file_name in yaml_files:
        with open(os.path.join(opt.evolve_population, file_name)) as yaml_file:
            value = yaml.safe_load(yaml_file)
            value = np.array([value[k] for k in hyp_GA.keys()])
            initial_values.append(list(value))

# Generate random values within the search space for the rest of the population
# 为种群中剩余的部分在搜索空间内生成随机值
if initial_values is None:
    population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size)]
elif pop_size > 1:
    population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size - len(initial_values))]
    for initial_value in initial_values:
        population = [initial_value] + population

# Run the genetic algorithm for a fixed number of generations
# 对固定的一代数运行遗传算法
list_keys = list(hyp_GA.keys())
for generation in range(opt.evolve):
    if generation >= 1:
        save_dict = {}
        for i in range(len(population)):
            little_dict = {list_keys[j]: float(population[i][j]) for j in range(len(population[i]))}
            save_dict[f"gen{str(generation)}number{str(i)}"] = little_dict

        with open(save_dir / "evolve_population.yaml", "w") as outfile:
            yaml.dump(save_dict, outfile, default_flow_style=False)

    # Adaptive elite size
    # 自适应精英的大小
    elite_size = min_elite_size + int((max_elite_size - min_elite_size) * (generation / opt.evolve))

    # Evaluate the fitness of each individual in the population
    # 评估种群中每个个体的适应度
    fitness_scores = []
    for individual in population:
        for key, value in zip(hyp_GA.keys(), individual):
            hyp_GA[key] = value
        hyp.update(hyp_GA)
        results = train(hyp.copy(), opt, device, callbacks)
        callbacks = Callbacks()
        # Write mutation results
        # 写入变异结果
        keys = (
            "metrics/precision",
            "metrics/recall",
            "metrics/mAP_0.5",
            "metrics/mAP_0.5:0.95",
            "val/box_loss",
            "val/obj_loss",
            "val/cls_loss",
        )
        print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)
        fitness_scores.append(results[2])

    # Select the fittest individuals for reproduction using adaptive tournament selection
    # 使用“自适应锦标赛选择”选择适应度最高的进行繁殖
    selected_indices = []
    for _ in range(pop_size - elite_size):
        # Adaptive tournament size
        # 自适应
        tournament_size = max(
            max(2, tournament_size_min),
            int(min(tournament_size_max, pop_size) - (generation / (opt.evolve / 10))),
        )
        # Perform tournament selection to choose the best individual
        # 执行锦标赛选择从而挑选出最佳的个体
        tournament_indices = random.sample(range(pop_size), tournament_size)
        tournament_fitness = [fitness_scores[j] for j in tournament_indices]
        winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        selected_indices.append(winner_index)

    # Add the elite individuals to the selected indices
    # 将精英个体添加到选定的索引中
    elite_indices = [i for i in range(pop_size) if fitness_scores[i] in sorted(fitness_scores)[-elite_size:]]
    selected_indices.extend(elite_indices)

    # Create the next generation through crossover and mutation
    # 通过交叉和变异创造下一代
    next_generation = []
    for _ in range(pop_size):
        parent1_index = selected_indices[random.randint(0, pop_size - 1)]
        parent2_index = selected_indices[random.randint(0, pop_size - 1)]

        # Adaptive crossover rate
        # 自适应交叉（交配）比例
        crossover_rate = max(
            crossover_rate_min, min(crossover_rate_max, crossover_rate_max - (generation / opt.evolve))
        )
        if random.uniform(0, 1) < crossover_rate:
            crossover_point = random.randint(1, len(hyp_GA) - 1)
            child = population[parent1_index][:crossover_point] + population[parent2_index][crossover_point:]
        else:
            child = population[parent1_index]

        # Adaptive mutation rate
        # 自适应变异比例
        mutation_rate = max(
            mutation_rate_min, min(mutation_rate_max, mutation_rate_max - (generation / opt.evolve))
        )
        for j in range(len(hyp_GA)):
            if random.uniform(0, 1) < mutation_rate:
                child[j] += random.uniform(-0.1, 0.1)
                child[j] = min(max(child[j], gene_ranges[j][0]), gene_ranges[j][1])
        next_generation.append(child)

    # Replace the old population with the new generation
    # 用新一代替换旧种群
    population = next_generation

# Print the best solution found
# 打印找到的最佳解决方案
best_index = fitness_scores.index(max(fitness_scores))
best_individual = population[best_index]
print("Best solution found:", best_individual)

# Plot results
# 绘制结果
plot_evolve(evolve_csv)
LOGGER.info(
    f'Hyperparameter evolution finished {opt.evolve} generations\n'
    f"Results saved to {colorstr('bold', save_dir)}\n"
    f'Usage example: $ python train.py --hyp {evolve_yaml}'
)
```

## 5.5 Automatic mixed precision (AMP) training

### 5.5.1 定义

自动混合精度（Automatic Mixed Precision, AMP）训练是一种深度学习训练技术，它可以在训练过程中动态地选择使用浮点数的精度。

自动混合精度训练的基本思想是，根据计算的需求和成本，自动地在单精度（FP32）和半精度（FP16）之间切换。具体来说，AMP 训练会识别那些对精度要求不高的计算（例如，权重矩阵的乘法），并将这些计算转换为半精度（FP16）计算，以减少梯度计算中的数值误差。而对于那些对精度要求较高的计算（例如，激活函数的计算），AMP 训练仍然使用单精度（FP32）计算，以保持模型的准确性和响应性。

> - float32: 单精度浮点数
> - float16: 半精度浮点数
> - float64: 双精度浮点数

### 5.5.2 AMP 训练的优点

1. **提高训练速度**：使用双精度进行某些计算可以减少浮点运算的次数，从而提高训练速度。
2. **减少内存使用**：双精度通常需要比单精度更多的内存，但只在必要时使用双精度，可以减少总体内存使用。
3. **提高数值稳定性**：在一些情况下，使用双精度可以减少梯度更新的数值误差，提高模型的训练稳定性。

### 5.5.3 `torch.FloatTensor` 和 `torch.HalfTensor`

在 PyTorch 中，`torch.FloatTensor` 和 `torch.HalfTensor` 是两种不同精度的浮点张量类型，它们分别对应于单精度（FP32）和半精度（FP16）浮点数。

- **torch.FloatTensor**：这是 PyTorch 中的单精度浮点张量。它使用 32 位（4 字节）来存储每个浮点数，提供了较高的数值精度和较大的数值范围。这是大多数深度学习任务中默认使用的浮点类型。
- **torch.HalfTensor**：这是 PyTorch 中的半精度浮点张量。它使用 16 位（2 字节）来存储每个浮点数，数值范围和精度都比单精度浮点数低。然而，由于半精度浮点数占用的内存较少，因此在某些情况下（如内存受限的环境或需要大幅提高计算速度时）会使用半精度浮点数。

### 5.5.4 作者答疑

在 [Automatic mixed precision (AMP) training is now natively supported and a stable feature. #557](https://github.com/ultralytics/yolov5/issues/557) 有提到 AMP。

<div align=center>
    <img src=./imgs_markdown/2024-02-07-15-05-30.png
    width=100%>
    <center></center>
</div>

从图中可以看到，💡 YOLOv5 默认开启 AMP 训练，并且保存的模型也是 FP16 而非传统的 FP32。

### 5.5.5 如何在 PyTorch 中使用 AMP？

> 💡 我之前写过相关博客：[PyTorch混合精度原理及如何开启该方法](https://blog.csdn.net/weixin_44878336/article/details/125433023)

```python
from torch.cuda.amp import Scaler, autocast
```

> ⚠️ 注意：
> 1. Scaler 并不是 AMP，autocast 也不是 AMP，只有 AMP + Scaler 才是 AMP
>
> 2. AMP 并不特指半精度，我们可以指定任意精度！

#### 5.5.5.1 autocast

- 〔官方文档〕[torch.cuda.amp.autocast](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast)

使用 `torch.cuda.amp` 模块中的 `autocast` 类。当进入 `autocast` 上下文后，支持 AMP 的 CUDA 算子会把 Tensor 的 `dtype` 转换为 FP16，从而在不损失训练精度的情况下加快运算。刚进入 `autocast` 的上下文时，Tensor 可以是任何类型，不需要在 `model` 或 `input` 上手工调用 `.half()`，框架会自动做，这就是 AMP 中的 Automatic。

另外需要注意的是，`autocast` 上下文应该只包含网络的前向推理过程（包括 loss 的计算），⚠️ 不要包含反向传播，因为 BP 的算子会使用和前向算子相同的类型。

---

我们看一下源码：

```python
class torch.autocast(device_type, 
                     dtype=None, 
                     enabled=True, 
                     cache_enabled=None)
```

**参数**：

- `device_type`（str，必需） - 要使用的设备类型。可能的值有：'cuda'，'cpu'，'xpu' 和 'hpu'。类型与 `torch.device` 的 `type` 属性相同。因此，我们可以使用 `Tensor.device.type` 获取张量的设备类型。
- `enabled`（bool，可选） - 区域内是否应启用 autocast。默认值：True
- `dtype`（torch_dtype，可选） - 是否使用 `torch.float16` 或 `torch.bfloat16`。
- `cache_enabled`（bool，可选） - 是否应启用 autocast 内部的权重缓存。默认值：True

> ⚠️ autocast 只是一个上下文管理器，会把在它范围内的 Tensor 的数据范围都统一，所以我们修改 `dtype` 参数来实现不同精度的计算，比如 `dtype=torch.float32, int8, ...`


`autocast` 的实例可用作上下文管理器或装饰器，允许脚本的某些区域以混合精度运行。

在这些区域中，操作以 `autocast` 选择的与操作特定的 `dtype` 运行，以提高性能同时保持准确性。

在进入启用 `autocast` 的区域时，张量可以是任何类型。在使用 autocasting 时，不应在模型或输入上调用 `half()` 或 `bfloat16()`。

`autocast` 应该仅包装网络的前向推理，包括损失计算。⚠️ 不建议在 autocast 下执行反向传递。反向操作在与 autocast 用于相应前向推理的相同类型中运行。

---

**CUDA 设备的示例**：

```python
# Creates model and optimizer in default precision
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

for input, target in data:
    optimizer.zero_grad()

    # Enables autocasting for the forward pass (model + loss)
    # 启用前向推理（模型 + 损失）的 autocast。
    with torch.autocast(device_type="cuda"):
        output = model(input)
        loss = loss_fn(output, target)

    # Exits the context manager before backward()
    # 在调用backward()之前退出上下文管理器。
    loss.backward()
    optimizer.step()
```

`autocast` 也可以作为装饰器使用，例如，可以应用在模型的 `forward` 方法上：

```python
class AutocastModel(nn.Module):
    ...
    @torch.autocast(device_type="cuda")
    def forward(self, input):
        ...
```

在启用了 `autocast` 的区域中产生的浮点张量可能是 `float16`（默认就是 FP16）。在返回到禁用 `autocast` 的区域后，将其与不同 `dtype` 的浮点张量一起使用可能导致类型不匹配错误。如果出现此情况，请将在 `autocast` 区域中生成的张量转回为 `float32`（或其他所需的 `dtype`）。如果 `autocast` 区域的张量已经是 `float32`，则转换是一个无操作，并且不会产生额外开销。

---

**CUDA 示例**：

```python
# Creates some tensors in default dtype (here assumed to be float32)
a_float32 = torch.rand((8, 8), device="cuda")
b_float32 = torch.rand((8, 8), device="cuda")
c_float32 = torch.rand((8, 8), device="cuda")
d_float32 = torch.rand((8, 8), device="cuda")

with torch.autocast(device_type="cuda"):
    # torch.mm is on autocast's list of ops that should run in float16.
    # torch.mm 在 autocast 的操作列表中，应该在 float16 中运行
    # Inputs are float32, but the op runs in float16 and produces float16 output.
    # 输入是 float32，但操作在 float16 中运行，并生成 float16 的输出
    # No manual casts are required.
    # 无需手动进行类型转换。
    e_float16 = torch.mm(a_float32, b_float32)

    # Also handles mixed input types
    # 还处理混合输入类型
    f_float16 = torch.mm(d_float32, e_float16)

# After exiting autocast, calls f_float16.float() to use with d_float32
# 在退出 autocast 后，调用 f_float16.float() 以与 d_float32 一起使用
g_float32 = torch.mm(d_float32, f_float16.float())  # 通过 .float() 将 FP16 转换为了 FP32
```

---

**CPU 训练示例**：

```python
# Creates model and optimizer in default precision
model = Net()
optimizer = optim.SGD(model.parameters(), ...)

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()

        # Runs the forward pass with autocasting.
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = model(input)
            loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()
```

#### 5.5.5.2 GradScaler

- 〔官方文档〕[torch.cuda.amp.GradScaler](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler)

使用 `torch.cuda.amp.GradScaler`，需要在训练最开始之前实例化一个 `GradScaler` 对象。通过放大 Loss 的值，从而防止梯度的 underflow（⚠️ 这只是 BP 的时候传递梯度信息使用，真正更新权重的时候还是要把放大的梯度再 unscale 回去）

---

我们看一下它的源码：

```python
class torch.cuda.amp.GradScaler(init_scale=65536.0, 
                                growth_factor=2.0, 
                                backoff_factor=0.5, 
                                growth_interval=2000, 
                                enabled=True)
```

**参数**：

- `init_scale`（float，可选，默认为 2.0**16） - 初始缩放因子。
- `growth_factor`（float，可选，默认为 2.0） - 如果在 `growth_interval` 连续的迭代中没有出现 inf/NaN 梯度，则在 `update()` 期间将缩放乘以此因子 —— **目的是尽最大可能将缩放因子变大**。
- `backoff_factor`（float，可选，默认为 0.5） - 如果在迭代中出现 inf/NaN 梯度，则在 `update()` 期间将缩放乘以此因子 —— 减小缩放因子避免模型无法训练。
- `growth_interval`（int，可选，默认为 2000） - 必须在没有 inf/NaN 梯度的连续迭代中发生的次数，以便通过 `growth_factor` 将缩放乘以此因子 —— 在 `growth_interval` 次迭代中都没有出现 inf/NaN 现象，就要放大缩放因子了。
- `enabled`（bool，可选） - 如果为 False，则禁用梯度缩放。`step()` 简单地调用底层的 `optimizer.step()`，而其他方法则成为无操作。默认值：True —— 提高兼容性用的

**方法**：

- `scaler.scale(loss)` 将给定的损失乘以缩放器当前的缩放因子。
- `scaler.step(optimizer)` 安全地取消缩放梯度并调用 `optimizer.step()`。
- `scaler.update()` 更新缩放器的缩放因子。

⚠️ 缩放因子通常会导致在前几次迭代中梯度中出现 infs/NaNs，因为其值进行校准。对于这些迭代，`scaler.step` 将跳过底层的 `optimizer.step()`。之后，跳过步骤应该很少发生（每几百或几千次迭代一次）。

#### 5.5.5.3 示例

##### 1. 典型的混合精度训练

```python
# Creates model and optimizer in default precision
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

# 在训练开始时创建一个 GradScaler 实例
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()  # 清空历史梯度

        # 使用 autocast 运行前向推理
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)

        # 缩放损失。对缩放后的损失调用 backward() 以创建缩放后的梯度。
        # ⚠️ 在 autocast 下执行反向传递是不推荐的
        # 在 autocast 选择的相应前向推理的 dtype 中运行反向操作
        scaler.scale(loss).backward()

        # scaler.step() 首先取消优化器的分配参数的梯度的缩放（从F32变为F16）
        # 如果这些梯度不包含无穷大或 NaN，然后调用 optimizer.step()
        # 否则，跳过 optimizer.step()
        scaler.step(optimizer)

        # 更新下一次迭代的缩放因子
        scaler.update()
```

##### 2. 梯度累积

梯度累积会将一个有效 Batch 大小（`batch_per_iter * iters_to_accumulate` * `num_procs`）内的梯度相加。缩放应该根据有效 Batch 进行校准，这意味着在有效 Batch 粒度上进行 inf/NaN 检查、如果发现 inf/NaN 梯度则跳过步骤，以及在有效 Batch 上更新缩放因子。而在给定有效 Batch 累积梯度期间，梯度应该保持缩放，缩放因子应该保持不变。如果在累积完成之前梯度被取消缩放（或缩放因子发生变化），那么下一次反向传递将会将缩放梯度添加到未缩放梯度中（或用不同因子缩放的梯度），之后就无法恢复累积的未缩放梯度，步骤必须应用。

因此，如果我们想要取消缩放梯度（例如，允许剪切未缩放梯度），请在执行步骤之前，在即将到来的步骤的所有（缩放的）梯度被累积后调用 `unscale_`。并且**只有在为一个完整的有效 Batch 调用了步骤的迭代结束时**才调用 `update`：

```python
scaler = GradScaler()

for epoch in epochs:
    for i, (input, target) in enumerate(data):
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)
            loss = loss / iters_to_accumulate

        # 累积缩放的梯度
        scaler.scale(loss).backward()

        if (i + 1) % iters_to_accumulate == 0:
            # 在这里可以使用 unscale_（如果需要），例如，允许剪切未缩放的梯度
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()  # 梯度清零需要放在最后了，不然梯度没法累积的
```

##### 3. 处理多个模型、损失和优化器

如果我们的网络有多个损失，我们必须对每个损失分别调用 `scaler.scale`。如果我们的网络有多个优化器，我们可以分别对每个优化器调用 `scaler.unscale_`，并且必须对每个优化器分别调用 `scaler.step`。

然而，⚠️ `scaler.update` 只应在此迭代中使用的所有优化器都已执行步骤之后调用一次：

```python
scaler = torch.cuda.amp.GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer0.zero_grad()
        optimizer1.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            output0 = model0(input)  # 第一个模型
            output1 = model1(input)  # 第二个模型
            loss0 = loss_fn(2 * output0 + 3 * output1, target)  # 混合损失1
            loss1 = loss_fn(3 * output0 - 5 * output1, target)  # 混合损失2

        # 这里的 retain_graph 与 amp 无关，它存在是因为在这个示例中，
        # 两个 backward() 调用共享了一些图的部分
        scaler.scale(loss0).backward(retain_graph=True)
        scaler.scale(loss1).backward()

        # 我们可以选择哪些优化器接收显式取消缩放，
        # 以便检查或修改它们拥有的参数的梯度。
        scaler.unscale_(optimizer0)

        scaler.step(optimizer0)
        scaler.step(optimizer1)

        scaler.update()
```

> ⚠️ 每个优化器都会检查其梯度中是否包含 inf/NaN，并独立决定是否跳过该步骤。这可能导致一个优化器跳过该步骤，而另一个不跳过。由于步骤跳过很少发生（每几百次迭代一次），这不应影响收敛性。

##### 4. DataParallel (DP) in a single process

即使 `torch.nn.DataParallel` 生成线程来在每个设备上运行前向推理，autocast 状态也会在每个线程中传播，以下操作将能够正常工作：

```python
model = MyModel()
dp_model = nn.DataParallel(model)

# 在主线程中设置 autocast
with autocast(device_type='cuda', dtype=torch.float16):
    # dp_model 内部的线程将使用 autocast。
    output = dp_model(input)
    # loss_fn 也使用 autocast
    loss = loss_fn(output)
```

##### 5. DistributedDataParallel (DDP), 单卡单线程

`torch.nn.parallel.DistributedDataParallel` 的文档建议每个进程使用一个 GPU 以获得最佳性能。在这种情况下，`DistributedDataParallel` 不会在内部生成线程，因此对 autocast 和 GradScaler 的使用不受影响。

##### 6. DistributedDataParallel (DDP), 多卡多线程

在这里，`torch.nn.parallel.DistributedDataParallel` 可能会生成一个辅助线程来在每个设备上运行前向推理，类似于 `torch.nn.DataParallel`。

解决方法是相同的：在模型的前向方法中应用 autocast，以确保它在辅助线程中启用。

## 5.6 断点续训

断点续训（Resume Training）是机器学习训练过程中的一个功能，它允许模型训练在之前停止的地方继续进行。这对于处理大型数据集或需要长时间训练的模型尤为重要，因为在训练过程中可能会由于各种原因（如硬件故障、电力中断等）导致训练过程意外停止。

在 YOLOv5 中实现断点续训通常涉及以下步骤：

1. **保存检查点（Checkpoint）**：在训练过程中，模型会定期保存检查点，这些检查点包含了模型参数、优化器状态以及当前的训练轮次等信息。
2. **中断训练**：如果训练过程中出现了中断，系统会停止更新这些检查点。
3. **恢复训练**：要恢复训练，用户需要指定上次保存的检查点文件。YOLOv5 训练脚本通常会有一个 `--resume` 参数，通过设置这个参数，可以从最近的检查点开始继续训练。
4. **设置**：在恢复训练之前，确保训练的设置（如学习率、批量大小、数据集等）与之前训练的设置保持一致，以确保训练过程的连续性和稳定性。
5. **继续训练**：启动训练脚本，程序会加载检查点文件，并从停止的地方开始继续训练模型。

断点续训不仅能够帮助节省时间，避免从头开始训练，还能够确保模型训练的连贯性和最终效果。在实际应用中，这是一个非常实用的功能，可以提高模型训练的效率。

我们看一下 `--resume` 在源码中的使用：

```python
parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")

...

def main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements(ROOT / "requirements.txt")

    # Resume (from specified or most recent last.pt)
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        # 先判断 opt.resume 是不是一个str，如果是，说明我们指定了具体的last.pt
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / "opt.yaml"  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors="ignore") as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location="cpu")["opt"]
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = "", str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
```

因为 `--resume` 是 `nargs="?"`，所以它可以有 0 个参数或者 1 个参数，即我们可以给它传参也可以不给它传参，那么它有如下两种用法：

```bash
# 用法1: 直接使用 last.pt 进行断点续训
python train.py --resume

# 用法2: 使用指定的权重进行断点续训
python train.py --resume runs/exp/weights/example_weights.pt
```

## 5.7 Multi-GPU Training，多 GPU 训练











# 参考

1. 〔视频教程〕[YOLOv5入门到精通！不愧是公认的讲的最好的【目标检测全套教程】同济大佬12小时带你从入门到进阶（YOLO/目标检测/环境部署+项目实战/Python/）](https://www.bilibili.com/video/BV1YG411876u?p=13)
2. 〔PyTorch 官方文档〕[torch.cuda.amp.autocast](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast)
3. 〔PyTorch 官方文档〕[torch.cuda.amp.GradScaler](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler)