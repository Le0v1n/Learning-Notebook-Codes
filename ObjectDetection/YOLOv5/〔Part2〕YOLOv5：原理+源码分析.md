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

        accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
        for j, x in enumerate(optimizer.param_groups):
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
            if "momentum" in x:
                x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])
```

💡 `numpy.interp(x, xp, fp, left=None, right=None, period=None)` 是 NumPy 中的一个函数，用于线性插值。线性插值是一种估算在两个已知值之间的未知值的方法，假设这些值之间的变化是线性的。

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

> ⚠️ x和y取的是索引，而xp和fp这里不是取索引，而是取值



