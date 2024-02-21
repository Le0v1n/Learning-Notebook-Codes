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
请根据我们的具体需求调整 `T_0` 和 `T_mult` 的值，以及 `num_epochs`，即我们的训练周期总数。

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
    if RANK in {-1, 0}:  # 判断是否在主线程中
        print_args(vars(opt))  # 打印所有参数
        check_git_status()  # 检查git的状态
        check_requirements(ROOT / "requirements.txt")  # 检查环境是否满足（如果不满足则自动安装）

    # Resume (from specified or most recent last.pt) | 断点续训
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        # 先判断 opt.resume 是不是一个str，如果是，说明我们指定了具体的last.pt
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())

        # 读取权重文件的上级上级文件夹下的opt.yaml文件
        opt_yaml = last.parent.parent / "opt.yaml"  # train options yaml

        # 将启动训练时的opt保存一下
        opt_data = opt.data  # original dataset

        if opt_yaml.is_file():  # 如果权重文件的上级上级文件夹下的opt.yaml文件存在
            with open(opt_yaml, errors="ignore") as f:
                d = yaml.safe_load(f)  # 加载所有的配置
        else:  # 如果不存在则读取权重中的opt
            d = torch.load(last, map_location="cpu")["opt"]

        # 将原来的opt使用读取到的opt进行覆盖
        opt = argparse.Namespace(**d)  # replace
        
        # 修改opt中的三个参数
        opt.cfg, opt.weights, opt.resume = "", str(last), True  # reinstate
```

因为 `--resume` 是 `nargs="?"`，所以它可以有 0 个参数或者 1 个参数，即我们可以给它传参也可以不给它传参，那么它有如下两种用法：

```bash
# 用法1: 直接使用 last.pt 进行断点续训
python train.py --resume

# 用法2: 使用指定的权重进行断点续训
python train.py --resume runs/exp/weights/example_weights.pt
```

## 5.7 Multi-GPU Training，多 GPU 训练

PyTorch 为数据并行训练提供了几种选项。对于从简单到复杂、从原型到生产逐渐增长的应用程序，常见的开发路径将是：

1. 〔**不使用 DDP**〕如果数据和模型可以适应一个 GPU，并且训练速度不是问题，那么我们直接使用单设备训练就行，不用 DDP。
2. 〔**单机多卡 - 不推荐**〕使用单机多 GPU 的 DataParallel，以最小的代码更改利用单台机器上的多个 GPU 来加快训练速度。
3. 〔**单机多卡 - 推荐**〕如果我们想进一步加快训练速度，并愿意编写更多代码，那么就使用单机多 GPU 的 DDP。
4. 〔**多机多卡**〕如果应用程序需要跨机器扩展，请使用多机器的 DistributedDataParallel 和启动脚本。
5. 〔**训练大模型**〕当数据和模型无法适应一个 GPU 时，在单机或多机上使用多 GPU 的 FullyShardedDataParallel (FSDP) 进行训练。
6. 〔**弹性训练**〕如果预期会出现错误（例如，内存不足）或者在训练过程中资源可以动态加入和离开，请使用 torch.distributed.elastic 启动分布式训练。

> 💡 DP 训练也适用于自动混合精度（AMP）

> 💡 之前写过一篇关于多 GPU 训练的博客：[PyTorch使用多GPU并行训练及其原理和注意事项](https://blog.csdn.net/weixin_44878336/article/details/125412625)

### 5.7.1 DP (Data Parallel，数据并行)

```python
class torch.nn.DataParallel(module, 
                            device_ids=None, 
                            output_device=None, 
                            dim=0)
```

可以看到，DP 可以在 module 级别实现数据并行。具体来说，这个容器（DP）通过在 Batch 维度上分块，将输入分割到指定的设备上，从而实现给定 module 的并行应用（其他对象将每个设备复制一次）。

- 在前向传播中， module 在每个设备上复制，并且每个副本处理输入的一部分。
- 在反向传播过程中，每个副本的梯度被汇总到原始 module 中。

需要注意的是：Batch 应该大于使用的 GPU 数量。

> ⚠️ 在 PyTorch 官方文档中也表明：建议使用 DistributedDataParallel 而不是这个类（DP）来进行多 GPU 训练，**即使只有一个节点**。

> DataParallel 包使单机多 GPU 并行变得非常容易，几乎不需要编写任何代码。它只需要对应用程序代码进行一次行更改。尽管 DataParallel 非常易于使用，但通常它的性能并不是最好的，因为它在每个前向传播中都复制了模型，并且它的单进程多线程并行自然会受到 GIL 争用的困扰。为了获得更好的性能，请考虑使用 DistributedDataParallel。


### 5.7.2 DDP（Distributed Data Parallel，分布式数据并行）

#### 5.7.2.1 DDP 介绍

Distributed Data Parallel (DDP) 是 PyTorch 中用于分布式训练的高级原语，它允许模型在多个节点上进行训练，每个节点可以有多个 GPU。DDP 可以显著提高训练速度，尤其是在使用大量数据和复杂模型时。

DDP 背后的主要思想是将模型复制到每个节点，并在每个节点上独立地处理一部分数据。在每个训练步骤中，每个节点上的模型副本会计算梯度，然后这些梯度会在所有节点之间进行平均。通过这种方式，DDP 可以确保每个节点上的模型参数保持同步。

DDP 相对于其他并行方法（如 DataParallel）的主要优势在于其高效的通信机制。DDP 使用 Ring Allreduce 算法来减少梯度同步时的通信瓶颈，这使得 DDP 特别适合于大型模型和大规模训练工作。

#### 5.7.2.2 DDP 的类定义

```python
class torch.nn.parallel.DistributedDataParallel(
        module,  # 要并行化的模块
        device_ids=None,  # device_ids必须用list
        output_device=None,
        dim=0, broadcast_buffers=True,
        process_group=None,
        bucket_cap_mb=25,
        find_unused_parameters=False,
        check_reduction=False,  # 此参数已弃用
        gradient_as_bucket_view=False,
        static_graph=False,
        delay_all_reduce_named_params=None,
        param_to_hook_all_reduce=None,
        mixed_precision=None,
        device_mesh=Non
)
```

#### 5.7.2.3 使用 DDP 的步骤

1. 初始化一个进程组，该进程组定义了参与训练的所有节点和它们之间的通信方式。
2. 将模型封装在 `DistributedDataParallel` 中，这样模型就可以在多个节点上并行训练。
3. 使用分布式 `Sampler` 确保每个节点只处理整个数据集的一部分，从而避免数据重复。
4. 在每个节点上运行训练循环，DDP 会自动处理梯度同步和模型更新。

DDP 还提供了一些其他功能，如自动故障恢复和模型检查点，这些功能对于长时间运行的大规模训练任务非常有用。

总的来说，DDP 是 PyTorch 中用于实现高效分布式训练的强大工具，它通过在多个节点上并行化模型和数据，使得训练大型模型变得更加可行和高效。

#### 5.7.2.4 DDP 的优点（相比于 DP）

与 DataParallel 相比，DistributedDataParallel 需要多一个步骤来设置，即调用 init_process_group。DDP 使用<font color='red'>多进程并行</font>，因此在模型副本之间没有 GIL 争用。此外，模型在 DDP 构建时进行广播，而不是在每次前向传播时，这也有助于加速训练。DDP 配备了多种性能优化技术。

> 💡 多线程会有 GIL 的问题，而多进程没有这种问题。

#### 5.7.2.5 通信协议介绍

在 PyTorch 的 `DistributedDataParallel`（DDP）中，通信协议主要指的是后端通信库，它用于在不同进程之间传输数据和梯度。截至我所知的信息，DDP 主要支持以下几种通信协议：

1. **GLOO**
   - 〔**特点**〕：GLOO 是 PyTorch 内置的通信库，支持 CPU 和 GPU 之间的通信。它适用于单个节点上的多 GPU 训练。
   - 〔**优点**〕：易于使用，实现简单，适合开发和测试。
   - 〔**缺点**〕：性能相对较低，<font color='red'><b>不适合大规模的分布式训练</b></font>。
2. **NCCL**
   - 〔**特点**〕：NCCL 是 NVIDIA Collective Communications Library 的缩写，专门为 NVIDIA GPU 设计的通信库。它支持 GPU 之间的通信，并且性能优秀。
   - 〔**优点**〕：<font color='red'><b>适用于大规模分布式训练，特别是在拥有大量 GPU 的场景中，能够提供高效的通信性能</b></font>。
   - 〔**缺点**〕：仅支持 NVIDIA GPU，需要在 NVIDIA 硬件上运行。
3. **MPI**
   - 〔**特点**〕：MPI 是 Message Passing Interface 的缩写，是一个跨语言的通信协议，广泛用于高性能计算。它支持在多节点之间进行通信。
   - 〔**优点**〕：非常成熟，支持大规模分布式训练，特别是在高性能计算环境中。
   - 〔**缺点**〕：需要外部依赖，配置和使用相对复杂，对开发者的要求较高。
4. **MPI_NCCL**
   - 〔**特点**〕：结合了 MPI 和 NCCL 的优点，可以在多节点之间进行 GPU 通信。
   - 〔**优点**〕：结合了 MPI 的分布式通信能力和 NCCL 在 GPU 之间的通信性能。
   - 〔**缺点**〕：实现相对复杂，需要同时配置 MPI 和 NCCL。
5. **TCP**
   - 〔**特点**〕：TCP 是传输控制协议，是一种广泛使用的网络通信协议。在 DDP 中，它通常用于在节点之间传输数据。
   - 〔**优点**〕：易于实现，兼容性好，适用于多种网络环境。
   - 〔**缺点**〕：性能相对较低，特别是在大规模分布式训练中可能成为瓶颈。

---

在选择通信协议时，需要根据具体的训练环境、硬件配置和性能需求来决定。例如：
- 〔**单机多卡**〕<font color='purple'><b></b>如果是在单个节点上有多个 GPU，GLOO 是一个不错的选择</font>。
- 〔**多机多卡**〕<font color='blue'><b>如果训练需要扩展到多个节点，NCCL 和 MPI 可能是更好的选择</b></font>。

#### 5.7.2.6 DDP 示例-1

让我们从一个简单的 `torch.nn.parallel.DistributedDataParallel` 示例开始。这个示例使用了一个 `torch.nn.Linear` 作为本地模型，用 DDP 包装它，然后在 DDP 模型上运行一次前向传播、一次反向传播和一个优化器步骤。之后，本地模型的参数将被更新，所有不同进程上的所有模型应该完全相同。

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
from torch.nn.parallel import DistributedDataParallel as DDP


def example(rank, world_size):
    # create default process group | 创建默认的进程组
    dist.init_process_group("gloo",  # 单机多卡推荐的协议
                            rank=rank,  # 表明目前是哪个进程
                            world_size=world_size  # 进程数量
                            )

    # create local model  | 创建本地模型
    model = nn.Linear(10, 10).to(rank)

    # construct DDP model | 构造DDP模型
    ddp_model = DDP(model, device_ids=[rank])  # device_ids必须用list

    # define loss function and optimizer | 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass  | 前向传播
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)

    # backward pass | 反向传播
    loss_fn(outputs, labels).backward()

    # update parameters | 更新参数
    optimizer.step()


def main():
    world_size = 2  # 进程数量
    # 使用PyTorch的多进程方法启动多进程
    mp.spawn(
        example,  # 进程执行的函数
        args=(world_size,),  # 进程执行函数的参数（不需要传递 rank 参数，mp.spawn函数会自动传入rank参数的）
        nprocs=world_size,  # 进程的数量
        join=True  # 进程等待（父进程是否应该等待子进程完成执行后再执行）
        )


if __name__=="__main__":
    # Environment variables which need to be set when using c10d's default "env" initialization mode.
    # 在使用 c10d 的默认 "env" 初始化模式时需要设置的环境变量
    os.environ["MASTER_ADDR"] = "localhost"  # 设置主节点（MasterNode）的IP为本地主机
    os.environ["MASTER_PORT"] = "29500"  # 设置主节点（MasterNode）的监听端口
    main()
```

> ⚠️ 在 `mp.spawn()` 函数中，`args=` 参数中不需要传递 `rank` 参数，`mp.spawn` 函数会自动传入rank参数的

---

<kbd><b>Question</b></kbd>：在 PyTorch 的 DistributedDataParallel 中，world_size 是什么？

<kbd><b>Answer</b></kbd>：在 PyTorch 的 `DistributedDataParallel`（DDP）中，`world_size` 是一个参数，它表示参与分布式训练的**总进程数**，即整个训练集群中的 GPU 数量。`world_size` 用于定义分布式环境中的并行级别，它告诉 DDP 有多少个进程（或 GPU）正在参与训练。

例如，如果你在一个拥有 8 个 GPU 的单个节点上训练，那么 `world_size` 就是 8。如果你有两个每个节点有 4 个 GPU 的节点，那么 `world_size` 就是 8。

> ⚠️ 在 DDP 中正确设置 `world_size` 非常重要，因为它会影响数据的分片方式、梯度的同步方式以及训练的整体流程。如果 `world_size` 设置不正确，可能会导致训练过程中的各种问题，如数据不均匀、梯度同步失败等。

---

<kbd><b>Question</b></kbd>：节点又是什么？<a id="explanation-node"></a>

<kbd><b>Answer</b></kbd>：在 PyTorch 的分布式训练中，<font color='red'>节点可以是一个物理服务器，也可以是一个虚拟机</font>，只要它能够运行 Python 程序并能够与网络中的其他节点通信。

在多节点分布式训练场景中，每个节点可能会有一个或多个 GPU，并且每个节点可能会运行一个或多个进程。每个节点上的进程可能会加载模型的一个副本，并处理数据集的一部分。这样，整个数据集就被分布在多个节点上，每个节点处理数据集的一个子集，从而实现数据并行和模型并行，以提高训练效率和处理能力。

在单机多 GPU 训练场景中，节点通常指的是单个物理服务器或工作站，它包含多个 GPU，并且这些 GPU 可以通过高速内部网络连接在一起，形成一个大的计算资源池。在这种情况下，节点内的 GPU 之间可以进行高效的通信和数据传输。

---

<kbd><b>Question</b></kbd>：进程为什么叫 world？

<kbd><b>Answer</b></kbd>：在分布式计算和并行训练的上下文中，术语 "world" 用于表示参与并行或分布式任务的全部进程集合。这种用法在各种并行和分布式系统的设计中都很常见，不仅仅限于 PyTorch 的 `DistributedDataParallel`（DDP）。

使用 "world" 来描述整个进程集合的概念源于这样的思想：在并行或分布式计算中，所有的计算资源（进程、节点、设备等）都被视为一个整体，它们共同协作来完成一个任务。这个整体就像是一个“世界”，其中的每个部分（进程）都是这个“世界”的一部分，它们之间需要协同工作，以确保整个任务的顺利进行。

在 PyTorch 的 DDP 中，`world_size` 参数就是用来表示这个“世界”中的进程数量，即参与训练的 GPU 数量。这个概念帮助开发者理解他们正在使用的是一个分布式的训练环境，其中的所有计算资源都是相互关联的，并且需要协同工作以提高训练效率。

所以，当你看到 `world_size` 这个词时，它通常指的是参与分布式训练的所有 GPU 或进程的总数。

---

<kbd><b>Question</b></kbd>：节点和进程之间的关系是什么？

<kbd><b>Answer</b></kbd>：在分布式计算和并行训练的上下文中，节点和进程之间的关系可以概括为：

1. **节点（Node）**：
   - 节点是物理或虚拟的计算设备，它可以是一个服务器、工作站或任何具有独立处理能力和网络连接的设备。
   - 在分布式系统中，节点是并行或分布式计算的基本单位。一个节点可以包含一个或多个处理器（如 CPU 或 GPU），并且**可以运行一个或多个进程**。
2. **进程（Process）**：
   - 进程是计算机中程序执行的基本单位。它是程序在执行过程中的一个实例，拥有独立的内存空间和执行线程。
   - **在分布式训练中，每个节点可以启动一个或多个进程**。**这些进程可以运行模型的副本，并处理数据集的一部分**。
   - 进程之间的通信和协作是实现分布式计算和并行处理的关键。

总结来说，节点是物理或虚拟的计算设备，而进程是这些节点上运行的程序实例。在分布式训练中，节点上可以启动多个进程，这些进程共同工作来处理数据并训练模型，每个进程负责模型的一部分或处理数据集的一部分。节点和进程之间的关系是并行和分布式系统设计的基础。

---

<kbd><b>Question</b></kbd>：rank 是什么？

<kbd><b>Answer</b></kbd>：在 PyTorch 的 `DistributedDataParallel`（DDP）中，`rank` 是指每个进程在分布式训练环境中的唯一标识符。在分布式训练中，每个节点可以启动多个进程，这些进程可能负责不同的任务，如数据加载、模型训练、参数同步等。<font color='red'>每个进程都有一个唯一的 `rank`，用于在分布式环境中进行识别和通信</font>。

在 DDP 中，`rank` 的值通常从 0 开始分配，每个节点的主进程通常会被分配一个 `rank`，而该节点的工作进程会被分配其他的 `rank`。💡 在多节点分布式训练中，所有节点的 `rank` 个数加起来就是 `world_size`。

> 💡 当需要同步模型参数时，只有 `rank=0` 的进程会收集其他所有进程的参数，然后广播给其他所有进程。
> 
> 💡 如果我们需要记录日志、print 一些东西的时候，我们也是会用 `rank=0` 的主线程来进行的，不然多线程同时进行的话会导致很多的重复！


下面我们画一张图来展示：

<div align=center>
    <img src=./imgs_markdown/plots-分布式训练的关系图.jpg
    width=80%>
    <center></center>
</div>

---

<kbd><b>Question</b></kbd>：`os.environ["MASTER_ADDR"] = "localhost"` 和 `os.environ["MASTER_PORT"] = "29500"` 的作用？

<kbd><b>Answer</b></kbd>：在 PyTorch 的分布式训练中，`MASTER_ADDR` 和 `MASTER_PORT` 环境变量用于指定主节点的地址和端口。这些变量在分布式训练的不同后端通信库中都有使用，包括 `gloo`、`nccl` 和 `mpi`。

具体来说，这两个环境变量的作用如下：

1. `MASTER_ADDR`：这个环境变量指定了主节点的 IP 地址或主机名。在单机多卡训练或多节点训练中，主节点是负责协调训练过程的进程。它通常会收集所有进程的梯度，并在所有进程之间同步模型参数。

2. `MASTER_PORT`：这个环境变量指定了主节点监听的端口。每个分布式训练的实例都需要一个唯一的端口来接收来自其他进程的连接请求。

在我们提供的代码中：

- `os.environ["MASTER_ADDR"] = "localhost"` 设置主节点的地址为本地主机，即当前运行代码的机器。
- `os.environ["MASTER_PORT"] = "29500"` 设置主节点监听的端口为 29500。

这些设置确保了分布式训练中的进程能够正确地连接到主节点，以便进行数据和梯度的同步。<font color='red'><b>如果这些环境变量设置不正确，分布式训练的进程可能无法相互通信，导致训练失败</b></font>。

---

<kbd><b>Question</b></kbd>：主节点是第一个开启的节点吗？

<kbd><b>Answer</b></kbd>：不，主节点并不一定是第一个开启的节点。在分布式训练中，主节点是负责协调训练过程的进程，它通常被分配一个特定的 `rank`，通常是 `rank 0`。主节点的角色是在训练过程中收集所有进程的梯度，并在所有进程之间同步模型参数。

主节点的选择通常是由训练代码中的逻辑决定的，而不是由启动顺序决定的。在 PyTorch 的 `DistributedDataParallel`（DDP）中，我们可以显式地指定哪个进程应该作为主节点。例如，我们可以使用 `dist.init_process_group` 函数并设置 `rank=0` 来指定主节点。

> ⚠️ <font color='red'><b>在多节点分布式训练中，每个节点上的主进程都可以是主节点，只要它们在启动时被正确地分配了 `rank 0`。这意味着，无论哪个节点首先启动，只要它的主进程被设置为 `rank 0`，它就可以成为主节点</b></font>。

总结来说，主节点是分布式训练中的一个关键进程，但它的选择并不依赖于它在网络中的启动顺序，而是依赖于训练代码中的设置和逻辑。

---

<kbd><b>Question</b></kbd>：主节点（Master Node）的主要任务是什么？

<kbd><b>Answer</b></kbd>：在分布式训练环境中，`rank 0` 的进程被指定为主节点（Master Node）。主节点负责以下任务：

1. **初始化**：在训练开始之前，主节点负责初始化分布式训练环境，例如设置网络通信和进程组。
2. **协调**：主节点负责协调分布式训练的各个进程，包括数据的划分、梯度的收集和参数的同步。
3. **监控**：主节点可以监控训练进度，并在训练完成后收集和保存模型。
4. **通信**：主节点是所有进程通信的中心点，负责接收来自其他进程的数据和梯度，并将其分发给所有进程。

在多节点分布式训练中，每个节点上通常会有一个主进程，它被设置为 `rank 0`。这意味着，每个节点都可以有一个主节点，负责管理该节点上的训练任务。

总之，`rank 0` 是一个特定的进程标识符，它在分布式训练中扮演着关键的角色，负责协调和管理工作。

#### 5.7.2.7 内部设计

本节通过深入探讨每一步骤的细节，揭示了 `torch.nn.parallel.DistributedDataParallel` 的工作原理。

〔**前提条件**〕DDP 依赖于 c10d ProcessGroup 进行通信。因此，应用程序必须在构建 DDP 之前创建 ProcessGroup 实例。

> `c10d` 是 PyTorch 中的一个库，它提供了集体通信（collective communication）的实现，这是分布式训练中用于进程间通信的一种机制。`c10d` 支持不同类型的进程组（ProcessGroup），这些进程组定义了进程之间的通信方式。
> 
> `ProcessGroup` 是 `c10d` 中的一个概念，它代表了一组通过某种通信协议连接在一起的进程。这些进程可以是在单个节点上的多个进程，也可以是跨多个节点的进程。`ProcessGroup` 负责管理进程之间的数据传输和同步。
> 
> 在 PyTorch 的 `DistributedDataParallel`（DDP）中，`ProcessGroup` 用于实现多 GPU 之间的数据并行训练。DDP 使用 `ProcessGroup` 来定义和维护分布式训练环境中的进程组，包括数据划分、梯度收集和参数同步等操作。
> 
> 总结来说，`c10d ProcessGroup` 是 PyTorch 中用于定义和实现分布式训练中进程间通信的一个抽象概念，它是 DDP 和 PyTorch 分布式训练中其他组件的基础。

---

〔**构建**〕DDP 构造函数（constructor）接受对本地模块的引用，并将 `rank=0` 进程的 `state_dict()` 广播到组中的所有其他进程，以确保所有模型副本从完全相同的状态开始。然后，每个 DDP 进程创建一个本地 Reducer，后者将在反向传播期间负责梯度的同步。

> 在 PyTorch 的 `DistributedDataParallel`（DDP）中，`Reducer` 是一个内部组件，用于在分布式训练过程中处理梯度的同步。`Reducer` 的主要职责是收集来自不同进程的梯度，然后对这些梯度进行聚合（例如，通过 allreduce 操作），以确保所有进程上的模型参数保持同步。
> 
> `Reducer` 的关键特点和功能包括：
> 1. **梯度聚合**：`Reducer` 负责将来自各个进程的梯度聚合为一个共享的梯度，这通常通过 allreduce 操作实现。
> 2. **分桶**：为了提高通信效率，`Reducer` 将参数梯度分桶（bucketing）。这意味着梯度根据它们的性质被组织到不同的桶中，每个桶中的梯度在同一时间内被聚合。
> 3. **异步通信**：`Reducer` 支持异步通信，<font color='red'><b>这意味着在等待一个桶的梯度聚合完成的同时，可以开始处理下一个桶的梯度</b></font>。
> 4. **自动求导钩子**：`Reducer` 在构建时注册自动求导（autograd）钩子，这些钩子在反向传播期间被触发，当梯度准备好时，它们会通知 `Reducer`。
> 5. **前向传播分析**：如果 `find_unused_parameters` 设置为 `True`，`Reducer` 会在前向传播期间分析模型输出，以确定哪些参数参与了反向传播。
> 
> 在 DDP 的训练循环中，`Reducer` 在反向传播期间起关键作用，它确保了所有进程上的模型参数在每次迭代后都是同步的。这种同步是通过在所有进程之间共享梯度来实现的，从而确保了模型的快速收敛和训练效率。

为了提高通信效率，`Reducer` 将参数梯度组织成桶中，并一次减少一个桶。可以通过在 DDP 构造函数中设置 `bucket_cap_mb` 参数来配置桶大小。参数梯度到桶的映射是在构建时确定的，基于桶大小限制和参数大小。模型参数按照大致上与给定模型的 `Model.parameters()` 相反的顺序分配到桶中。使用相反顺序的原因是 DDP 期望在反向传播期间梯度按照大约那个顺序准备好。下图显示了一个示例。

<div align=center>
    <img src=./imgs_markdown/2024-02-18-13-50-04.png
    width=90%>
    <center></center>
</div>

注意，`grad0` 和 `grad1` 在 `bucket1` 中，另外两个梯度在 `bucket0` 中。当然，这个假设可能并不总是正确的，当发生这种情况时，它可能会降低 DDP 反向速度，因为 `Reducer` 不能在最早可能的时间启动通信。

除了分桶之外，`Reducer` 在构建时还注册 `autograd` 钩子，每个参数一个。这些钩子将在反向传播期间梯度准备好时触发。

---

〔**前向传播**〕DDP 接受输入并传递给本地模型，然后分析本地模型的输出，如果 `find_unused_parameters` 设置为 `True`。这种模式允许在模型的子图上运行反向传播，并且 DDP 通过遍历自动微分（autograd）图从模型输出**找出哪些参数参与了反向传播**，并标记所有未使用的参数为 【ready for reduction】。在反向传播过程中，Reducer 只会等待未准备好的参数，但仍会对所有桶进行求和。将参数梯度标记为准备好目前并不帮助 DDP 跳过桶，但它将防止 DDP 在反向传播期间永远等待缺失的梯度。

> ⚠️ 请注意，遍历自动微分图会引入额外的开销，因此应在必要时将 `find_unused_parameters` 设置为 `True`，否则设置为 `False`。

---

〔**反向传播**〕`.backward()` 函数直接在损失 Tensor 上被调用，这超出了 DDP 的控制范围，DDP 使用在构建时注册的 autograd 钩子来触发梯度同步。当一个梯度准备好时，其对应的 DDP 钩子在该梯度累积器上触发，DDP 然后将该参数梯度标记为 【ready for reduction】。

当一个桶中的所有梯度都准备好时，Reducer 启动对该桶的异步 allreduce，以计算所有进程上梯度的平均值。

当所有桶都准备好时，Reducer 将等待所有 allreduce 操作完成。完成此操作后，平均梯度被写入所有参数的 `param.grad` 字段。因此，在反向传播之后，不同 DDP 进程上对应参数的 grad 字段应该相同。

---

〔**优化器步骤**〕从优化器的角度来看，它正在优化一个本地模型。DDP 进程上的所有模型副本可以保持同步，因为它们都从相同的状态开始，并且在每次迭代中都有相同的平均梯度。

> ⚠️ DDP 要求所有进程中都有 Reducer 实例，以便以完全相同的顺序调用 allreduce，这是通过始终按桶索引顺序而不是实际的桶准备顺序运行 allreduce 来实现的。进程之间 allreduce 顺序的不匹配可能导致错误的结果或 DDP 反向传播挂起。

#### 5.7.2.8 DDP 与 DP 的对比

在我们深入了解之前，让我们先搞清楚为什么尽管增加了复杂性，我们仍然会考虑使用 DDP 而不是 DP：

首先，DP 是单进程、<font color='red'><b>多线程</b></font>的，并且只能在单机上工作，而 DDP 是<font color='red'><b>多进程</b></font>的，并且既支持单机也支持多机训练。<u>即使在单机上，DP 通常也比 DDP 慢</u>，这主要是因为线程间的 GIL 争用、每迭代复制的模型，以及输入分散和输出聚集引入的额外开销。

回想一下前面的教程，如果我们的模型太大，无法适应单个 GPU，我们必须使用模型并行将其拆分到多个 GPU 上。DDP 支持模型并行，但 DP 目前不支持。当 DDP 与模型并行结合使用时，每个 DDP 进程将使用模型并行，所有进程集体将使用数据并行。

|方式|数据并行|模型并行|
|:-:|:-:|:-:|
|DP|✔️|❌|
|DDP|✔️|✔️|

#### 5.7.2.9 DDP 示例-2

```python
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    # rank: 进程索引
    # world_size: 进程总量

    os.environ['MASTER_ADDR'] = 'localhost'  # 设置主节点（MasterNode）的IP为本地主机
    os.environ['MASTER_PORT'] = '12355'  # 设置主节点（MasterNode）的监听端口

    # initialize the process group | 初始化进程组
    dist.init_process_group(backend="gloo", 
                            rank=rank, 
                            world_size=world_size)

def cleanup():
    dist.destroy_process_group()  # 关闭进程组
```

需要注意的是：在 Windows 平台上，`torch.distributed` 包仅支持 `Gloo` `后端、FileStore` 和 `TcpStore。`
对于 `FileStore`，请在 `init_process_group` 中的 `init_method` 参数中设置为`本地文件路径`。示例如下：

```python
init_method = "file:///f:/libtmp/some_file"
dist.init_process_group(
   "gloo",
   rank=rank,
   init_method=init_method,
   world_size=world_size)
```

对于 TcpStore，在 Windows 上的设置方式与 Linux 相同。

现在，我们创建一个示例模型（ToyModel），用 DDP 包装它，并向其提供一些模拟输入数据。

> ⚠️ 请注意，由于 DDP 在构造函数中从 rank 0 进程向所有其他 DDP 进程广播模型状态，因此我们不需要担心不同的 DDP 进程从不同的初始模型参数值开始。

```python
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    # 创建模型并使用DDP将其送到对应的GPU中
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])  # device_ids必须用list
    print(f"[{rank}] 模型已创建并使用 DDP 进行了包装...")

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()  # 清空历史梯度信息
    outputs = ddp_model(torch.randn(20, 10))  # 前向推理
    labels = torch.randn(20, 5).to(rank)  # 创建GT
    loss_fn(outputs, labels).backward()  # 计算损失
    optimizer.step()  # 更新参数
    print(f"[{rank}] 模型运行完毕")

    cleanup()  # 关闭进程组
    print(f"[{rank}] 关闭进程组!")


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
    
if __name__ == "__main__":
    run_demo(demo_fn=demo_basic, world_size=4)
```

结果如下：

```
Running basic DDP example on rank 0.
Running basic DDP example on rank 2.
Running basic DDP example on rank 3.
Running basic DDP example on rank 1.
[2] 模型已创建并使用 DDP 进行了包装...
[0] 模型已创建并使用 DDP 进行了包装...
[3] 模型已创建并使用 DDP 进行了包装...
[1] 模型已创建并使用 DDP 进行了包装...
[3] 模型运行完毕
[1] 模型运行完毕
[3] 关闭进程组!
[1] 关闭进程组!
[2] 模型运行完毕
[2] 关闭进程组!
[0] 模型运行完毕
[0] 关闭进程组!
```

我们可以看到，进程的开始和结束并不是统一的，所以我们一般是 `rank=0` 这个主节点进行 print 操作，即：

```python
def demo_basic(rank, world_size):
    # print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    # 创建模型并使用DDP将其送到对应的GPU中
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])  # device_ids必须用list
    print(f"模型已创建并使用 DDP 进行了包装...") if rank == 0 else ...

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()  # 清空历史梯度信息
    outputs = ddp_model(torch.randn(20, 10))  # 前向推理
    labels = torch.randn(20, 5).to(rank)  # 创建GT
    loss_fn(outputs, labels).backward()  # 计算损失
    optimizer.step()  # 更新参数
    print(f"模型运行完毕") if rank == 0 else ...

    cleanup()  # 关闭进程组
    print(f"关闭进程组!") if rank == 0 else ...


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
    
if __name__ == "__main__":
    run_demo(demo_fn=demo_basic, world_size=4)
```

```
模型已创建并使用 DDP 进行了包装...
模型运行完毕
关闭进程组!
```

> ⚠️ 在DDP中，构造函数、前向传播和反向传播都是分布式同步点。不同进程应该启动相同数量的同步，并以相同的顺序到达这些同步点，并且**大致**同时进入每个同步点。否则，快速进程可能会提前到达并在等待掉队者时超时。因此，用户负责平衡跨进程的工作负载分布。有时，由于网络延迟、资源争用或不可预测的工作负载峰值等原因，处理速度偏差是不可避免的。为了在这些情况下避免超时，请确保在调用 `init_process_group` 时传递一个足够大的超时值。

```python
class dist.init_process_group(backend=None, 
                              init_method=None, 
                              timeout=None,  # 超时值
                              world_size=-1, 
                              rank=-1, 
                              store=None, 
                              group_name='', 
                              pg_options=None)
```

`timeout`：针对进程组执行的操作的超时时间。NCCL 的后端默认值为 10 分钟，其他后端默认值为 30 分钟。这是集体操作将在之后被异步中止，并且进程将崩溃的持续时间。这样做是因为 CUDA 执行是异步的，一旦异步 NCCL 操作失败，继续执行用户代码将不再安全，因为失败的异步 NCCL 操作可能导致后续的 CUDA 操作在损坏的数据上运行。当设置 `TORCH_NCCL_BLOCKING_WAIT` 时，进程将阻塞并等待这个超时时间。

#### 5.7.2.10 Save and Load Checkpoints，保存和加载检查点

在训练过程中，通常会使用 `torch.save` 和 `torch.load` 来检查点模块，并从检查点恢复。当使用 DDP 时，<font color='red'>一种优化方法是在一个进程中保存模型，然后将其加载到所有进程中，从而减少写操作的开销</font>。这是正确的，**因为所有进程都从相同的参数开始，并且在反向传播中同步梯度，因此优化器应该继续将参数设置为相同的值**。如果我们使用这种优化，请确保在保存完成之前没有进程开始加载。<font color='red'><b>此外，在加载模块时，我们需要提供一个适当的 map_location 参数，以防止进程进入其他进程的设备</b></font>。如果缺少 `map_location`，`torch.load` 将首先将模块加载到 CPU，然后将每个参数复制到它保存的位置，这将导致同一台机器上的所有进程使用同一组设备。

```python
def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)  # 初始化进程组

    model = ToyModel().to(rank)  # 定义模型
    ddp_model = DDP(model, device_ids=[rank])  # device_ids必须用list

    # ckpt保存路径
    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    
    if rank == 0:
        # All processes should see same parameters as they all start from same random parameters and gradients are synchronized in backward passes. Therefore, saving it in one process is sufficient.
        # 所有进程应该看到相同的参数，因为它们都从相同的随机参数开始，并且在反向传播中同步梯度。因此，在一个进程中保存就足够了
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process 0 saves it.
    # ⚠️ 使用barrier()来确保进程1在进程0保存模型之后加载模型
    dist.barrier()
    
    # configure map_location properly | 正确配置map_location
    # ⚠️ %0确保了所有进程在加载模型时都首先将其加载到cuda:0设备上，然后根据每个进程的rank将其参数移动到正确的设备上。
    # 这样可以避免在加载模型时出现设备冲突
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    print(f"[rank={rank}]map_location: {map_location}")
    weights = torch.load(CHECKPOINT_PATH, map_location=map_location)
    ddp_model.load_state_dict(weights)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)

    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below as the AllReduce ops in the backward pass of DDP already served as a synchronization.
    # 在DDP的后向传播中的AllReduce操作已经起到了同步作用，因此不需要使用dist.barrier()来保护下面的文件删除操作
    if rank == 0:
        os.remove(CHECKPOINT_PATH)  # 删除掉示例中的权值文件

    cleanup()
```

```
Running DDP checkpoint example on rank 3.
Running DDP checkpoint example on rank 2.
Running DDP checkpoint example on rank 0.
Running DDP checkpoint example on rank 1.
[rank=2]map_location: {'cuda:0': 'cuda:2'}
[rank=0]map_location: {'cuda:0': 'cuda:0'}
[rank=1]map_location: {'cuda:0': 'cuda:1'}
[rank=3]map_location: {'cuda:0': 'cuda:3'}
```

<kbd><b>Question</b></kbd>：`dist.barrier()` 的作用？

<kbd><b>Answer</b></kbd>：`dist.barrier()` 是 PyTorch 中分布式训练中的一个同步操作。它的作用是在分布式环境中的多个进程之间创建一个同步点，确保所有进程都在此同步点到达之前等待。

具体来说，`dist.barrier()` 的作用包括：

1. **同步点：** 当某个进程调用 `dist.barrier()` 时，它会被阻塞，直到所有参与分布式训练的进程都到达这个同步点。

2. **确保一致性：** 在某些情况下，你可能希望在所有进程都完成了某个任务之后再继续进行下一步操作。`dist.barrier()` 提供了一种确保所有进程都达到某个状态后再继续执行的机制。

注意事项：
- 使用 `dist.barrier()` 时，确保所有参与分布式训练的进程都调用了该函数，否则可能会导致死锁。
- 尽量在合适的地方使用 `dist.barrier()`，以避免不必要的等待时间。

示例：

```python
import os
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    
    
def example(rank, world_size):
    setup(rank=rank, world_size=world_size)  # 初始化进程组

    print(f"Before barrier [rank {rank}]")
    dist.barrier()
    print(f"After barrier [rank {rank}]")
    
    # 关闭进程组
    dist.destroy_process_group()
    

def main():
    world_size = 4
    
    mp.spawn(
        fn=example,
        args=(world_size, ),  # 不需要传递 rank 参数，mp.spawn函数会自动传入rank参数的
        nprocs=world_size,
        join=True
    )
    
    
if __name__ == "__main__":
    main()
```

```
Before barrier [rank 0]
Before barrier [rank 2]
Before barrier [rank 1]
Before barrier [rank 3]

⚠️ 这里会等待一会儿再打印出下面的内容

After barrier [rank 1]
After barrier [rank 2]
After barrier [rank 3]
After barrier [rank 0]
```

在上面的示例中，每个进程在调用 `dist.barrier()` 之前打印一条消息，然后在调用之后再打印一条消息。你会注意到在所有进程都调用 `dist.barrier()` 之前，没有任何一个进程能够打印 "After barrier" 的消息。这展示了 `dist.barrier()` 的同步效果。

#### 5.7.2.11 将 DDP 与模型并行（MP）结合（Combining DDP with Model Parallelism）

DDP 也适用于多 GPU 模型。当训练具有大量数据的大型模型时，使用 DDP 包装多 GPU 模型特别有帮助。

```python
class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0  # 设备0
        self.dev1 = dev1  # 设备1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)  # 将特征图放到设备0上
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)  # 将特征图放到设备1上
        return self.net2(x)
```

当将多 GPU 模型传递给 DDP 时，不应设置 `device_ids` 和 `output_device`。输入和输出数据将由应用程序或模型的 `forward()` 方法放置在适当的设备上。

```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # 设置主节点（MasterNode）的IP为本地主机
    os.environ['MASTER_PORT'] = '12355'  # 设置主节点（MasterNode）的监听端口
    
    dist.init_process_group(backend='gloo',
                            world_size=world_size,
                            rank=rank)


def cleanup():
    dist.destroy_process_group()  # 关闭进程组


class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0  # 设备0
        self.dev1 = dev1  # 设备1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)  # 将特征图放到设备0上
        print(f"已加载到设备-{self.dev0}")
        x = self.relu(self.net1(x))

        x = x.to(self.dev1)  # 将特征图放到设备1上
        print(f"已加载到设备-{self.dev1}")
        return self.net2(x)
    
    
def demo_model_parallel(rank, world_size):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    # setup mp_model and devices for this process
    # 确保每个进程处理两个相邻的 GPU 设备
    dev0 = rank * 2  # 奇数
    dev1 = rank * 2 + 1  # 偶数
    print(f"[rank{rank}] {dev0=}    {dev1=}")
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()
    
    
def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn, 
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()  # 计算GPU个数
    print(f"{n_gpus = }")
    
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    
    # world_size = n_gpus
    world_size = n_gpus // 2  # 因为涉及到了模型并行MP，这里我们是一个模型被分为了两部分，所以就是一个线程负责模型的两部分，需要两个GPU
    run_demo(demo_model_parallel, world_size)
```

```
n_gpus = 8

Running DDP with model parallel example on rank 1.
Running DDP with model parallel example on rank 2.
Running DDP with model parallel example on rank 0.
Running DDP with model parallel example on rank 3.

[rank0] dev0=0    dev1=1
[rank3] dev0=6    dev1=7
[rank2] dev0=4    dev1=5
[rank1] dev0=2    dev1=3

已加载到设备-0
已加载到设备-4

已加载到设备-6
已加载到设备-2

已加载到设备-7
已加载到设备-3

已加载到设备-5
已加载到设备-1
```

#### 5.7.2.12 使用 torch.distributed.run/torchrun 初始化 DDP

我们可以利用 PyTorch Elastic 来简化 DDP 代码，更容易地初始化作业。让我们仍然使用 Toymodel 示例，并创建一个名为 `exp5-elastic_ddp.py` 的文件。

```python
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic():
    dist.init_process_group("nccl")
    rank = dist.get_rank()  # 获取线程rank
    # print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    print(f"{device_id = }\t{rank = }")
    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])  # device_ids必须用list

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_id)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    dist.destroy_process_group()


if __name__ == "__main__":
    demo_basic()
```

然后我们可以在所有节点上运行 `torch elastic` 或者 `torchrun` 命令来初始化上述创建的 DDP 作业：

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

# 方法1
torchrun --nproc_per_node 7 --master_port 34231 learn-20240218/exp5-elastic_ddp.py

# 方法2
python -m torch.distributed.launch --nproc_per_node 7 --master_port 34231 learn-20240218/exp5-elastic_ddp.py
```

```
device_id = 0   rank = 0
device_id = 2   rank = 2
device_id = 4   rank = 4
device_id = 5   rank = 5
device_id = 3   rank = 3
device_id = 1   rank = 1
device_id = 6   rank = 6
```

〔**命令解析**〕：
- `--nproc_per_node=7`: 此参数指定每个节点上的进程数。在这里，设置为 7，意味着每个节点上将有 7 个进程，如果是一个 GPU 一个进程，那么就意味着有 7 个 GPU。
- `--master_port=34231`: 主节点（MasterNode）监听端口

#### 5.7.2.13 单机模型并行（Model Parallel，MP）练习

模型并行（Model Parallel，MP）在分布式训练技术中得到了广泛应用。之前的内容解释了如何使用 DataParallel 在多个 GPU 上训练神经网络，这个特性将同一个模型复制到所有 GPU 上，每个 GPU 处理输入数据的不同的分区。尽管它可以显著加速训练过程，但它不适用于一些场景，在这些场景中，模型太大，无法放入单个 GPU 中。

接下来将展示如何使用模型并行（MP）来解决该问题，与 DataParallel 不同，MP 将单个模型分割到不同的 GPU 上，而不是在每个 GPU 上复制整个模型（具体来说，假设一个模型 m 包含 10 层：当使用 DataParallel 时，每个 GPU 都将拥有这些 10 层的副本，而当在两个 GPU 上使用 MP 时，每个 GPU 可能托管 5 层）。

MP 的高级思想是将模型的不同子网络放置在不同的设备上，并相应地实现 `forward` 方法，以在不同设备之间移动中间输出。由于只有模型的部分在单个设备上运行，一组设备可以共同处理更大的模型。在本文中，我们不会尝试构建巨大的模型并将它们挤压到有限数量的 GPU 上。相反，本文的重点是展示模型并行的想法。将想法应用到实际应用中取决于读者。

〔**简单示例**〕

让我们从一个包含两个线性层的玩具模型开始。为了在两个 GPU 上运行这个模型，只需将每个线性层放在不同的 GPU 上，并相应地移动输入和中间输出以匹配层设备。

```python
import torch
import torch.nn as nn
import torch.optim as optim


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10).to('cuda:0')
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to('cuda:1')

    def forward(self, x):
        x = self.relu(self.net1(x.to('cuda:0')))
        return self.net2(x.to('cuda:1'))
```

注意，上述 ToyModel 看起来与如何在单个 GPU 上实现它非常相似，除了四个 `.to(device)` 调用，这些调用将线性层和张量放置在适当的设备上。这是模型中唯一需要更改的地方。`.backward()` 和 `torch.optim` 将自动处理梯度，就像模型在单个 GPU 上一样。我们只需要确保在调用损失函数时，标签位于与输出相同的设备上。

```python
model = ToyModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

optimizer.zero_grad()
outputs = model(torch.randn(20, 10))
labels = torch.randn(20, 5).to('cuda:1')  # 要与 output 的设备类型统一
loss_fn(outputs, labels).backward()
optimizer.step()
```

〔**将模型并行应用于现有模块**〕

我们还可以仅通过几行代码更改，将现有的单 GPU 模块运行在多个 GPU 上。下面的代码展示了如何将 `torchvision.models.resnet50()` 分解到两个 GPU 上。想法是继承现有的 ResNet 模块，并在构造过程中将层分割到两个 GPU 上。然后，覆盖 forward 方法，通过相应地移动中间输出将两个子网络拼接在一起。

```python
import time
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck, resnet50


class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=1000, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')

        self.fc.to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        return self.fc(x.view(x.size(0), -1))
    
    
if __name__ == "__main__":
    
    # warm-up
    model = resnet50().to(0)
    del model
    
    # MP
    t1 = time.time()
    model = ModelParallelResNet50()
    t2 = time.time()
    print(f"MP: {t2 - t1 = }")
    
    # release gpu
    del model

    # without MP
    t1 = time.time()
    model = resnet50().to(0)
    t2 = time.time()
    print(f"No MP: {t2 - t1 = }")
```

```
MP:    t2 - t1 = 3.3101274967193604
No MP: t2 - t1 = 0.39362549781799316
```

上述实现解决了模型太大无法放入单个 GPU 的情况。然而如果模型适合单个 GPU，它的速度会比在单个 GPU 上运行慢。这是因为，**在任何时刻，只有两个 GPU 中的一个在工作，而另一个则无所事事**。随着中间输出需要在层 2 和层 3 之间从 cuda:0 复制到 cuda:1，性能进一步恶化。

让我们进行一个实验，以获得更定量的执行时间视图。在这个实验中，我们通过运行随机输入和标签来训练 ModelParallelResNet50 和现有的 torchvision.models.resnet50()。训练结束后，模型不会产生任何有用的预测，但我们可以对执行时间有一个合理的了解。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.resnet import ResNet, Bottleneck
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import timeit


class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=1000, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')

        self.fc.to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        return self.fc(x.view(x.size(0), -1))


def train(model):
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    one_hot_indices = torch.LongTensor(batch_size).random_(0, num_classes).view(batch_size, 1)

    for _ in range(num_batches):
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes).scatter_(1, one_hot_indices, 1)

        # run forward pass
        optimizer.zero_grad()
        outputs = model(inputs.to('cuda:0'))

        # run backward pass
        labels = labels.to(outputs.device)
        loss_fn(outputs, labels).backward()
        optimizer.step()
        
        
def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
        align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('ResNet50 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)
        

if __name__ == "__main__":
    num_batches = 3
    batch_size = 120
    image_w = 128
    image_h = 128
    num_classes = 1000

    num_repeat = 10

    stmt = "train(model)"

    setup = "model = ModelParallelResNet50()"
    mp_run_times = timeit.repeat(stmt, setup, number=1, repeat=num_repeat, globals=globals())
    mp_run_times = mp_run_times[1:]  # 舍弃第一次的结果
    mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)
    print(f"{mp_mean = :.4f}")
    print(f"{mp_std = :.4f}\n")

    setup = "import torchvision.models as models;" + "model = models.resnet50(num_classes=num_classes).to('cuda:0')"
    rn_run_times = timeit.repeat(stmt, setup, number=1, repeat=num_repeat, globals=globals())
    rn_run_times = rn_run_times[1:]
    rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)
    print(f"{rn_mean = :.4f}")
    print(f"{rn_std = :.4f}")

    plot([mp_mean, rn_mean],
        [mp_std, rn_std],
        ['Model Parallel', 'Single GPU'],
        'mp_vs_rn.png')
```

```
mp_mean = 0.5374
mp_std = 0.0056

rn_mean = 0.5002
rn_std = 0.0039
```

<div align=center>
    <img src=./imgs_markdown/2024-02-18-17-35-19.png
    width=65%>
    <center></center>
</div>

上面的 `train(model)` 方法使用 `nn.MSELoss` 作为损失函数，`optim.SGD` 作为优化器。它模拟了对 128x128 图像的训练，这些图像被组织成 3 个 Batch，每个 Batch 包含 120 张图像。然后，我们使用 `timeit` 运行 `train(model)` 方法 10 次，并绘制执行时间及其标准差。

结果显示，模型并行实现的执行时间比现有的单 GPU 实现长了 (0.5374/0.5002-1)*100=7.44%。因此，我们可以得出结论，在 GPU 之间复制张量的大致开销为 7%。有改进的空间，因为我们都知道在整个执行过程中，两个 GPU 中的一个处于闲置状态。一个选项是将每个 Batch 进一步划分为一系列的分割，这样当一个分割到达第二个子网络时，接下来的分割可以被输入到第一个子网络。通过这种方式，两个连续的分割可以在两个 GPU 上同时运行。

> 在上述描述中，提到的“分割”（splits）是一种假设的模型并行策略，用于优化模型并行训练中的 GPU 利用率。这种策略的核心思想是将每个训练批次（Batch）进一步细分成多个更小的部分，并确保这些部分在两个 GPU 之间均匀分布。
> 
> 这里是一个简化的例子来说明这个策略：
> 
> 假设我们有一个批次（Batch）包含 128 个图像，并且我们有两个 GPU。如果我们按照传统的模型并行方法，我们可能会将前 64 个图像分配给 GPU 0，将后 64 个图像分配给 GPU 1。这样，在处理完前 64 个图像并将其结果发送到 GPU 1 之后，GPU 0 会等待 GPU 1 处理完后 64 个图像。这种情况下，GPU 0 会有一段时间处于闲置状态。
> 
> 为了解决这个问题，我们可以将每个 Batch 进一步划分为多个“分割”（splits）。例如，我们可以将每个 Batch 划分为 4 个分割，每个分割包含 32 个图像。这样，每个 GPU 都会同时处理两个分割。当一个分割在 GPU 0 上处理完并到达第二个子网络时，GPU 1 已经处理完其第一个分割，并准备接收 GPU 0 的第二个分割。通过这种方式，两个 GPU 可以更有效地并行工作，减少闲置时间。
> 
> 请注意，这只是一个示例，实际的分割策略可能会根据模型的结构和数据的特点而有所不同。这种策略的关键是确保每个 GPU 都有连续的输入数据处理，从而减少 GPU 之间的等待时间，提高训练效率。
>
> 上述的方式如下图所示：
>
> <div align=center>
>     <img src=./imgs_markdown/plots-MP.jpg
>     width=100%>
>     <center></center>
> </div>

〔**通过流水线输入 (Pipelining Inputs) 加速**〕

在接下来的实验中，我们将每个包含 120 张图像的 Batch 进一步划分为 60 张图像的分割。由于 PyTorch 以异步方式启动 CUDA 操作，实现不需要 `spawn` 多个线程来达到并发性。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.resnet import ResNet, Bottleneck
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import timeit
import prettytable


class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=1000, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')

        self.fc.to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        return self.fc(x.view(x.size(0), -1))
    
    
class PipelineParallelResNet50(ModelParallelResNet50):
    """这个类使用了流水线并行技术，将输入数据分割成更小的部分，并在两个不同的GPU上并行处理。
        seq1和seq2是ResNet50模型的不同部分，它们在不同的GPU上运行。
        通过这种方式，可以最大化GPU的利用率和计算效率。
    """
    def __init__(self, split_size=60, *args, **kwargs):
        super(PipelineParallelResNet50, self).__init__(*args, **kwargs)
        self.split_size = split_size  # 设置split_size参数，用于将输入数据分割成更小的部分

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))  # 沿着batch维度进行split，并使其变为可迭代对象
        s_next = next(splits)  # 获取下一个分割的数据
        s_prev = self.seq1(s_next).to('cuda:1')  # 将分割的数据通过seq1处理，并将结果发送到cuda:1
        ret = []  # 创建一个空列表，用于存储处理结果

        for s_next in splits:
            # A. ``s_prev`` runs on ``cuda:1``
            s_prev = self.seq2(s_prev)  # 在cuda:1上处理s_prev
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))  # 将处理结果通过全连接层，并将结果添加到ret列表中

            # B. ``s_next`` runs on ``cuda:0``, which can run concurrently with A
            s_prev = self.seq1(s_next).to('cuda:1')  # 在cuda:0上处理s_next，并将结果发送到cuda:1

        s_prev = self.seq2(s_prev)  # 处理最后一个分割的数据
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))  # 将处理结果添加到ret列表中

        return torch.cat(ret)  # 将所有处理结果拼接起来，并返回


def train(model):
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    one_hot_indices = torch.LongTensor(batch_size).random_(0, num_classes).view(batch_size, 1)

    for _ in range(num_batches):
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes).scatter_(1, one_hot_indices, 1)

        # run forward pass
        optimizer.zero_grad()
        outputs = model(inputs.to('cuda:0'))

        # run backward pass
        labels = labels.to(outputs.device)
        loss_fn(outputs, labels).backward()
        optimizer.step()
        
        
def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
        align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('ResNet50 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)
        

if __name__ == "__main__":
    num_batches = 3
    batch_size = 120
    image_w = 128
    image_h = 128
    num_classes = 1000
    num_repeat = 10
    
    ptable = prettytable.PrettyTable(["model", 'mean', 'std'])
    ptable.border = True

    stmt = "train(model)"

    setup = "model = ModelParallelResNet50()"
    mp_run_times = timeit.repeat(stmt, setup, number=1, repeat=num_repeat, globals=globals())[1:]  # 舍弃第一次的结果
    mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)
    ptable.add_row(['Model Parallel', round(mp_mean, 4), round(mp_std, 4)])

    setup = "import torchvision.models as models;" + "model = models.resnet50(num_classes=num_classes).to('cuda:0')"
    rn_run_times = timeit.repeat(stmt, setup, number=1, repeat=num_repeat, globals=globals())[1:]
    rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)
    ptable.add_row(['Single GPU', round(rn_mean, 4), round(rn_std, 4)])
    
    setup = "model = PipelineParallelResNet50()"
    pp_run_times = timeit.repeat(stmt, setup, number=1, repeat=num_repeat, globals=globals())[1:]
    pp_mean, pp_std = np.mean(pp_run_times), np.std(pp_run_times)
    ptable.add_row(['Pipelining Model Parallel', round(pp_mean, 4), round(pp_std, 4)])
    
    print(ptable)

    plot([mp_mean, rn_mean, pp_mean],
        [mp_std, rn_std, pp_std],
        ['Model Parallel', 'Single GPU', 'Pipelining Model Parallel'],
        'mp_vs_rn_vs_pp.png')
```

```
+---------------------------+--------+--------+
|           model           |  mean  |  std   |
+---------------------------+--------+--------+
|       Model Parallel      | 0.5319 | 0.0077 |
|         Single GPU        | 0.4971 | 0.0048 |
| Pipelining Model Parallel | 0.5103 | 0.0099 |
+---------------------------+--------+--------+
```

<div align=center>
    <img src=./imgs_markdown/2024-02-19-10-28-26.png
    width=80%>
    <center></center>
</div>

实验结果表明，将输入数据流水线化到模型并行的 ResNet50 可以加快训练过程，大约提高了 4.2328%（0.5319/0.5103-1）。这与 PyTorch 官方提供的结果相差有点大。

> <div align=center>
>     <img src=./imgs_markdown/2024-02-19-10-30-00.png
>     width=80%>
>     <center>PyTorch 官方提供的结果</center>
> </div>
>
> 在 PyTorch 官方提供的结果中，使用 Pipline 的 MP 提升了 49% 的速度，但在我们的机器上并没有得出对应的结论（RTX 2080Ti）。


由于在我们的流水线并行实现中引入了一个新的参数 `split_size`，目前尚不清楚这个新参数如何影响整体训练时间。直观地说，使用较小的 `split_size` 会导致许多微小的 CUDA 内核启动，而使用较大的 `split_size` 会在第一个和最后一个 split 期间导致相对较长的空闲时间。这两种情况都不理想。可能存在一个针对这个特定实验的最优 `split_size` 配置。让我们通过使用几个不同的 `split_size` 值进行实验来尝试找到它。

```python
...

means = []
stds = []
split_sizes = [1, 3, 5, 8, 10, 12, 20, 40, 60, 80, 100]

# 创建表格对象
table = prettytable.PrettyTable()
table.field_names = ["split_size", "mean", "std"]  # 添加列名
table.border = True  # 设置表格格式

for split_size in split_sizes:
    setup = "model = PipelineParallelResNet50(split_size=%d)" % split_size
    pp_run_times = timeit.repeat(stmt, setup, number=1, repeat=num_repeat, globals=globals())[1:]  # 去掉第一个结果以减少误差
    means.append(np.mean(pp_run_times))
    stds.append(np.std(pp_run_times))
    # print(f"[split_size={split_size}]\n\tmean: {means[-1]:.4f}\n\tstd: {stds[-1]:.4f}")
    table.add_row([split_size, round(means[-1], 4), round(stds[-1], 4)])  # 添加行数据
    print(table)

fig, ax = plt.subplots()
ax.plot(split_sizes, means)
ax.errorbar(split_sizes, means, yerr=stds, ecolor='red', fmt='ro')
ax.set_ylabel('ResNet50 Execution Time (Second)')
ax.set_xlabel('Pipeline Split Size')
ax.set_xticks(split_sizes)
ax.yaxis.grid(True)
plt.tight_layout()
plt.savefig("split_size_tradeoff.png")
plt.close(fig)
```

```
+------------+--------+--------+
| split_size |  mean  |  std   |
+------------+--------+--------+
|     1      | 7.2939 | 0.1057 |
|     3      | 2.7698 | 0.0721 |
|     5      | 1.7975 |  0.02  |
|     8      | 1.273  |  0.02  |
|     10     | 1.0167 | 0.0288 |
|     12     | 0.8842 | 0.0442 |
|     20     | 0.7015 | 0.0263 |
|     40     | 0.5397 | 0.0103 |
|     60     | 0.5086 | 0.0157 |
|     80     | 0.5375 | 0.0113 |
|    100     | 0.5584 | 0.0048 |
+------------+--------+--------+
```

<div align=center>
    <img src=./imgs_markdown/2024-02-19-10-22-14.png
    width=80%>
    <center></center>
</div>

结果显示，将 `split_size` 设置为 60 可以实现最快的训练速度，这导致了 54%（3.75/2.43-1）的速度提升。仍然有机会进一步加速训练过程。例如，所有在 cuda:0 上的操作都被放置在其默认流上。这意味着下一个 split 的计算无法与上一个 split 的复制操作重叠。然而，由于 prev 和 next splits 是不同的张量，重叠一个的计算与另一个的复制是没有问题的。实现需要在两个 GPU 上使用多个流，并且不同的子网络结构需要不同的流管理策略。

> ⚠️ 该结果也与 PyTorch 官方提供的结果有差异，PyTorch 官方提供的结果如下：
> 
> <div align=center>
>     <img src=./imgs_markdown/2024-02-19-10-32-05.png
>     width=80%>
>     <center>PyTorch 官方提供的结果</center>
> </div>
> 
> 在该结果中，`split_size` 的最佳结果为 12，而我们的最佳结果是 60。

💡 这篇文章展示了几种性能测量。当我们在自己的机器上运行相同代码时，可能会看到不同的数字，因为结果取决于底层硬件和软件。为了获得我们环境下的最佳性能，一个合适的方法是首先生成曲线以确定最佳的分割大小，然后使用该分割大小来流水线化输入。

## 5.8 Parallelizing Data Loading 并行化数据加载

<div align=center>
    <img src=./imgs_markdown/plots-数据加载.jpg
    width=100%>
    <center></center>
</div>

---

〔**DP流程**〕

1. Transfer minibatch data from page-locked memory to GPU 0 (master). Master GPU also holds the model. Other GPUs have a stale copy of the model.
    将小批量 (minibatch) 数据从页锁定内存传输到 GPU 0（主 GPU）。主 GPU 还保存着模型。其他 GPU 则保存着模型的旧副本
2. Scatter minibatch data across GPUs
    在多个 GPU 上分散小批量数据
3. Replicate model across GPUs
    在多个 GPU 上复制模型
4. Run forward pass on each GPU, compute output. PyTorch implementataion spins up separate threads to parallelize forward pass
    在每个 GPU 上运行前向传播，计算输出。PyTorch 实现会启动 (spin up) 单独的线程来并行化前向传播
5. Gather ouput on master GPU, compute loss
    在主 GPU 上收集输出，计算损失
6. Scatter loss to GPUs and run backward pass to calculate parameter gradients
    在 GPU 上分散损失并运行反向传播以计算参数梯度
7. Reduce gradients on GPU 0
    在 GPU 0 上减少梯度
8. Update model's parameters
    更新模型的参数

---

〔**DDP流程**〕

1. Load data from disk into page-locked memory on the host. Use multiple worker processes to parallelize data load. Distributed mini-batch sampler ensures that each process loads non-overlapping data
    将数据从磁盘加载到主机上的页锁定内存中。使用多个工作进程来并行加载数据。**分布式小批量采样器确保每个进程加载不重叠的数据**
2. Trasfer mini-batch data from page-locked memory to each GPU concurrently. No data broadcast is needed. Each GPU has an identical copy of the model and no model broadcast is needed either
    同时将小批量数据从页锁定内存传输到每个 GPU。无需数据广播。每个 GPU 都有模型的相同副本，也无需模型广播
3. Run forward pass on each GPU, compute output
    在每个 GPU 上运行前向传播，计算输出
4. Compute loss, run backward pass to compute gradients. Perform gradient all-reduce in parallel with gradient computation
    计算损失，运行反向传播以计算梯度。在与梯度计算并行的情况下执行梯度全归约
5. Update model's parameters. Because each GPU started with identical copy of the model and gradients were all-reduced, weights updates on all GPUs are identical. Thus no model sync is required.
    更新模型的参数。<font color='red'><b>由于每个 GPU 都是从相同的模型副本开始的，并且梯度已经全归约，所以所有 GPU 上的权重更新都是相同的。因此，不需要模型同步</b></font>

## 5.9 YOLOv5 开启 GPU 训练

> 相关 Issue：[Multi-GPU Training 🌟 #475](https://github.com/ultralytics/yolov5/issues/475)

### 5.9.1 ✔️〔推荐〕单 GPU 训练

```bash
python train.py \
    --weights weights/yolov5s.pt \
    --data data/coco128.yaml \
    --hyp data/hyps/hyp.scratch-low.yaml \
    --epochs 150 \
    --batch-size 64 \
    --imgsz 640 \
    --project runs/train \
    --name exp
    --device 0
```

### 5.9.2 ⚠️〔不推荐〕DP 训练

```bash
python train.py \
    --weights weights/yolov5s.pt \
    --data data/coco128.yaml \
    --hyp data/hyps/hyp.scratch-low.yaml \
    --epochs 150 \
    --batch-size 64 \
    --imgsz 640 \
    --project runs/train \
    --name exp \
    --device 0,1
```

### 5.9.3 ✔️〔推荐〕DDP 训练

```bash
python -m torch.distributed.run \
    --nproc_per_node 4 \  # 每个节点的 GPU 数量
    train.py \
    --weights weights/yolov5s.pt \
    --data data/coco128.yaml \
    --hyp data/hyps/hyp.scratch-low.yaml \
    --epochs 150 \
    --batch-size 64 \
    --imgsz 640 \
    --project runs/train \
    --name exp \
    --device 0,1,2,3
```

其中，`--nproc_per_node` 表明一个节点中 GPU 的数量

> 关于节点是什么，我们在这里[node的说明](#explanation-node)进行了说明。



# 参考

1. 〔视频教程〕[YOLOv5入门到精通！不愧是公认的讲的最好的【目标检测全套教程】同济大佬12小时带我们从入门到进阶（YOLO/目标检测/环境部署+项目实战/Python/）](https://www.bilibili.com/video/BV1YG411876u?p=13)
2. 〔PyTorch 官方文档〕[torch.cuda.amp.autocast](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast)
3. 〔PyTorch 官方文档〕[torch.cuda.amp.GradScaler](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler)
3. 〔PyTorch 官方文档〕[PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)