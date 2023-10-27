<center><b><font size=12>YOLOv5：参数说明</font></b></center>

本文主要探索 YOLOv5 参数。

<kbd>Key Words</kbd>：YOLOv5、参数、早停、自动Batch、cache、device、seed、single_cls、single class、rect、rectangular、letterbox、

# 1. Arguments

YOLO 模型的训练设置指的是用于在数据集上训练模型的各种超参数和配置。这些设置可能会影响模型的性能、速度和准确性。一些常见的 YOLO 训练设置包括批量大小、学习率、动量和权重衰减。可能会影响训练过程的其他因素包括优化器的选择、损失函数的选择以及训练数据集的大小和组成。重要的是要仔细调整和尝试这些设置，以在特定任务中实现最佳性能。

| Key | Value | Description |
|:- |:- |:- |
| <kbd>model</kbd> | `None` | 模型文件的路径，例如 `yolov8n.pt`，`yolov8n.yaml` |
| <kbd>data</kbd> | `None` | 数据文件的路径，例如 `coco128.yaml` |
| <kbd>epochs</kbd> | `100` | 训练的轮数 |
| <kbd>patience</kbd> [^footnote-early-stop] | `50` | 用于早停训练的等待轮数 |
| <kbd>batch</kbd> [^footnote-batch] | `16` | 每个 Batch 中图像的数量（`-1` 为自动批处理）|
| <kbd>imgsz</kbd> | `640` | 输入图像的尺寸，以整数表示 |
| <kbd>save</kbd> | `True` | 保存训练检查点和预测结果 |
| <kbd>save_period</kbd> | `-1` | 多少轮保存一次模型（如果小于 1 则禁用）|
| <kbd>cache</kbd> [^footnote-cache] | `False` | 是否使用缓存进行数据加载（可选：`ram` / `disk`）|
| <kbd>device</kbd> [^footnote-device] | `None` | 运行的设备，例如 `device=0` 或 `device=0,1,2,3` 或 `device=cpu` |
| <kbd>workers</kbd> | `8` | 数据加载的工作线程数量（每个 RANK 如果 DDP）|
| <kbd>project</kbd> | `None` | 项目名称 |
| <kbd>name</kbd> | `None` | 实验名称 |
| <kbd>exist_ok</kbd> | `False` | 是否覆盖现有实验 |
| <kbd>pretrained</kbd> | `True` | 是否使用预训练模型（bool）或要加载权重的模型（str）|
| <kbd>optimizer</kbd> [^footnote-optimizer] | `'auto'` | 要使用的优化器，可选项有 <br>[`'SGD'`、`'Adam'`、`'AdamW'`、`'RMSProp'`</br>`'NAdam'`、`'RAdam'`、`'auto'`、`'Adamax'`] |
| <kbd>verbose</kbd> | `False` | 是否打印详细输出 |
| <kbd>seed</kbd> [^footnote-seed] | `0` | 用于可重现性的随机种子 |
| <kbd>deterministic</kbd> [^footnote-deterministic] | `True` | 是否启用确定性模式 |
| <kbd>single_cls</kbd> [^footnote-single-cls]  | `False` | 表明只有一个类别（train multi-class data as single-class）|
| <kbd>rect</kbd> [^footnote-rect] | `False` | 使用最小填充的每批矩形训练 |
| <kbd>cos_lr</kbd> | `False` | 是否使用余弦学习率调度 |
| <kbd>close_mosaic</kbd> | 10 | 禁用最后几轮的马赛克增强（0 禁用）|
| <kbd>resume</kbd> | `False` | 从最后一个检查点恢复训练 |
| <kbd>amp</kbd> | True | 自动混合精度（AMP）训练，可选项: `[True, False]` |
| <kbd>fraction</kbd> | 1.0 | 要训练的数据集比例（默认为 1.0，训练集中的所有图像）|
| <kbd>profile</kbd> | `False` | 在训练期间为记录器启用 ONNX 和 TensorRT 速度 |
| <kbd>freeze</kbd> | `None` | 冻结前 `n` 层（`int` 或 `list`，可选）或在训练期间冻结的层索引列表 |
| <kbd>lr0</kbd> | `0.01` | 初始学习率（例如 `SGD=1E-2`，`Adam=1E-3`）|
| <kbd>lrf</kbd> | `0.01` | 最终学习率（`lr0 * lrf`）|
| <kbd>momentum</kbd> | `0.937` | SGD 动量/Adam beta1 |
| <kbd>weight_decay</kbd> | `0.0005` | 优化器权重衰减 5e-4 |
| <kbd>warmup_epochs</kbd> | `3.0` | 热身的 Epoch 数（允许分数）|
| <kbd>warmup_momentum</kbd> | `0.8` | 热身初始动量 |
| <kbd>warmup_bias_lr</kbd> | `0.1` | 热身初始偏置 lr |
| <kbd>box</kbd> [^footnote-box-loss] | `7.5` | Box 损失增益（定位损失权重） |
| <kbd>cls</kbd> [^footnote-cls-loss] | `0.5` | cls 损失增益（与像素一起缩放）（类别损失权重）|
| <kbd>dfl</kbd> [^footnote-dfl-loss] | `1.5` | dfl 损失增益 |
| <kbd>pose</kbd> | `12.0` | 姿势损失增益（仅姿势）|
| <kbd>kobj</kbd> | `2.0` | 关键点对象损失增益（仅姿势）|
| <kbd>label_smoothing</kbd> | `0.0` | 标签平滑（分数）|
| <kbd>nbs</kbd> | `64` | 名义批量大小 |
| <kbd>overlap_mask</kbd> | `True` | 训练期间掩码应该重叠（仅分割训练）|
| <kbd>mask_ratio</kbd> | `4` | 掩码下采样比率（仅分割训练）|
| <kbd>dropout</kbd> | `0.0` | 使用 dropout 正则化（仅分类训练）|
| <kbd>val</kbd> | `True` | 训练期间验证/测试 |

[^footnote-early-stop]: 请见 [Early Stop，早停](#early-stop早停)
[^footnote-batch]: 请见 [batch_size=-1，自动决定 batch size 大小](#batch_size-1自动决定-batch-size-大小)
[^footnote-cache]: 请见 [cache](#拓展-cache)
[^footnote-device]: 请见 [device](#拓展-device)
[^footnote-optimizer]: 看了源码，发现只有 `Adam` `AdamW` `RMSProp` `SGD`，剩下都没有 🙃
[^footnote-seed]: 请见 [seed](#拓展-seed)
[^footnote-deterministic]: 请见 [seed](#拓展-seed)
[^footnote-single-cls]: 请见 [Single Class](#拓展-single-class)
[^footnote-rect]: 请见 [矩阵训练(rectangular training)](#拓展-矩阵训练rectangular-training)
[^footnote-box-loss]: 请见 [损失函数权重](#拓展-损失函数权重)
[^footnote-cls-loss]: 请见 [损失函数权重](#2-cls-损失权重)
[^footnote-dfl-loss]: 请见 [损失函数权重](#3-dfldistribution-focal-loss损失权重)

## <kbd>拓展</kbd> Early Stop，早停

在 YOLO 中，"early stop"（早停）是一种训练过程的策略，旨在在训练模型时在达到一定条件时提前终止训练，以避免过度拟合或浪费计算资源。早停的核心思想是根据某个性能指标或损失函数的变化来监控模型的训练进展，并在其开始出现下降趋势的时候终止训练，以防止模型在继续训练时过度拟合训练数据。

在 YOLO 中，早停通常涉及以下步骤：

1. **选择性能指标或损失函数：** 首先，需要选择一个性能指标（如验证集上的准确度）或损失函数（如验证集上的损失值）来监测模型的性能。这个指标通常与训练的任务和目标有关。

2. **设定早停条件：** 接下来，定义早停条件。通常，早停条件是当监测的指标或损失函数在一连续的一定数量的轮次中不再有明显的改善时触发早停。例如，可以设定一个"耐心值"，如果指标在连续的若干轮次内没有显著提高，就触发早停。

3. **监控训练过程：** 在模型训练期间，定期检查所选性能指标或损失函数的值。如果该值在一段时间内没有明显改善，就开始考虑终止训练。

4. **终止训练：** 一旦早停条件满足，训练过程就会被终止，模型的当前状态将被保存。这通常是为了避免进一步训练导致模型在验证集上的性能下降。

早停是一种有效的训练策略，可以帮助防止模型在训练期间过度拟合，并节省训练时间和计算资源。通过监控性能指标，可以在模型达到最佳性能时停止训练，而不是持续训练直到出现过拟合。这有助于提高模型的泛化能力。

在 YOLOv5 中，早停的代码如下：

```python
class EarlyStopping:
    # YOLOv5 simple early stopper
    def __init__(self, patience=30):
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                        f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n'
                        f'To update EarlyStopping(patience={self.patience}) pass a new patience value, '
                        f'i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.')
        return stop
```

这是早停（early stop）的简单实现，以下是对代码的分析：

1. `__init__` 方法：这是类的构造函数，用于初始化 `EarlyStopping` 对象。它接受一个参数 `patience`，该参数表示等待多少轮没有性能改善时停止训练。默认情况下，`patience` 被设置为无穷大，表示不启用早停。

2. 类属性：
   - `best_fitness`：用于存储当前已观察到的最佳性能值（通常是模型的平均精度 mAP）。
   - `best_epoch`：记录达到最佳性能值的 Epoch。
   - `patience`：指定允许多少个 Epoch 后没有性能改善时触发早停。默认值是无穷大。
   - `possible_stop`：一个布尔值，表示在下一轮可能触发早停。

3. `__call__` 方法：这是一个可调用的方法，用于在每个训练轮次中更新早停的状态并检查是否应该停止训练。它接受两个参数：`epoch` 表示当前轮次，`fitness` 表示当前的性能值。

   - 首先，它检查当前性能是否 $\ge$ 最佳性能值，如果是，则更新最佳性能值和最佳轮次。
   - 然后，它计算当前 Epoch 与最佳 Epoch 的差距 `delta`。
   - 接下来，它检查是否下一 Epoch 可能触发早停，即 `delta` 是否大于等于`patience - 1`。
   - 最后，它检查是否应该停止训练，即 `delta` 是否大于等于 `patience`。
   - 如果需要早停，它会打印一条信息，指出没有在最近的一段时间内观察到性能改善，同时提供了最佳模型保存的信息。

这个类的主要作用是跟踪模型的性能，并根据早停条件决定是否停止训练。当性能在一定轮次内没有显著提高时，它将触发早停，以避免过度拟合。早停的条件可以通过 `patience` 参数来调整。如果需要更长的训练，可以增加 `patience` 的值；如果要禁用早停，可以将 `patience` 设置为 0。

在 YOLOv5 中，该类的对象创建在 `train.py/train函数` 中：

```python
def train(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp 
    ...  # 省略
    dictionarystopper, stop = EarlyStopping(patience=opt.patience), False
```

在一次 Epoch 后，会判断 `stop` 决定是否早停：

```python
        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks
```

## <kbd>拓展</kbd> batch_size=-1，自动决定 batch size 大小

在 YOLOv5 中，如果设置 `--batch-size -1`，则程序会在 `train.py` 中的 `train` 函数中调用下面代码：

```python
# Batch size
if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
    batch_size = check_train_batch_size(model, imgsz, amp)
    loggers.on_params_update({'batch_size': batch_size})
```

从而实现自动计算 `batch_size`。那我们看一下 `check_train_batch_size` 这个函数：

```python
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Auto-batch utils
"""

from copy import deepcopy
import numpy as np
import torch

from utils.general import LOGGER, colorstr
from utils.torch_utils import profile

def check_train_batch_size(model, imgsz=640, amp=True):
    # 检查 YOLOv5 训练批处理大小
    with torch.cuda.amp.autocast(amp):
        return autobatch(deepcopy(model).train(), imgsz)  # 计算最佳批处理大小

def autobatch(model, imgsz=640, fraction=0.8, batch_size=16):
    # 自动估算最佳的 YOLOv5 批处理大小，以使用可用 CUDA 内存的 `fraction`
    # 用法:
    #     import torch
    #     from utils.autobatch import autobatch
    #     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False)
    #     print(autobatch(model))  # 可以看到，autobatch 可以wrap模型

    # 检查设备
    prefix = colorstr('AutoBatch: ')
    LOGGER.info(f'{prefix}计算 --imgsz {imgsz} 的最佳批处理大小')
    device = next(model.parameters()).device  # 获取模型的设备
    if device.type == 'cpu':
        LOGGER.info(f'{prefix}未检测到 CUDA，使用默认的 CPU 批处理大小 {batch_size}')
        return batch_size
    if torch.backends.cudnn.benchmark:
        LOGGER.info(f'{prefix} ⚠️ 需要禁用 torch.backends.cudnn.benchmark=False，使用默认批处理大小 {batch_size}')
        return batch_size

    # 检查 CUDA 内存
    gb = 1 << 30  # 字节转 GiB (1024 ** 3)
    d = str(device).upper()  # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)  # 设备属性
    t = properties.total_memory / gb  # GiB 总内存
    r = torch.cuda.memory_reserved(device) / gb  # GiB 保留内存
    a = torch.cuda.memory_allocated(device) / gb  # GiB 分配内存
    f = t - (r + a)  # GiB 空闲内存
    LOGGER.info(f'{prefix}{d} ({properties.name}) {t:.2f}G 总内存, {r:.2f}G 保留, {a:.2f}G 分配, {f:.2f}G 空闲')

    # 分析批处理大小
    batch_sizes = [1, 2, 4, 8, 16]
    try:
        img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]
        results = profile(img, model, n=3, device=device)
    except Exception as e:
        LOGGER.warning(f'{prefix}{e}')

    # 拟合解决方案
    y = [x[2] for x in results if x]  # 内存 [2]
    p = np.polyfit(batch_sizes[:len(y)], y, deg=1)  # 一次多项式拟合
    b = int((f * fraction - p[1]) / p[0])  # y 截距 (最佳批处理大小)
    if None in results:  # 一些大小失败
        i = results.index(None)  # 第一个失败索引
        if b >= batch_sizes[i]:  # y 截距在失败点之上
            b = batch_sizes[max(i - 1, 0)]  # 选择前一个安全点
    if b < 1 or b > 1024:  # b 超出安全范围
        b = batch_size
        LOGGER.warning(f'{prefix}WARNING ⚠️ CUDA 异常检测，建议重新启动环境并重试命令。')

    fraction = (np.polyval(p, b) + r + a) / t  # 预测的实际分数
    LOGGER.info(f'{prefix}为 {d} 使用批处理大小 {b} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅')
    return b
```

简单来说，就是通过计算内存和显存从而获取最佳的 batch size 大小。

## <kbd>拓展</kbd> cache

YOLOv5 的 cache 在 `utils/dataloaders.py/class LoadImagesAndLabels(Dataset):` 中，代码太长了，也懒得看了。反之是用 `numpy` 和 `pickle` 进行的缓存读取，可以将缓存放在 RAM 或 硬盘中。

在训练中使用缓存（cache）功能可以带来一些好处，尤其是在处理大规模数据集时。以下是使用缓存的一些主要好处：

1. **加速数据加载**：使用缓存可以将数据加载到内存或更快的存储介质中，从而加速数据读取速度。这对于大型数据集来说特别有用，因为数据加载通常是训练过程中的瓶颈之一。

2. **降低 I/O 负担**：数据集通常存储在磁盘上，每次从磁盘读取数据都会引入 I/O 操作，这是相对较慢的。通过将数据缓存到内存或更快的存储设备中，可以大大减轻 I/O 负担，提高数据读取效率。

3. **稳定训练速度**：在训练过程中，数据加载速度可能会波动，导致训练速度不稳定。使用缓存可以减少这种波动，使训练速度更加一致，有助于更可靠地控制训练进程。

4. **节省资源**：一旦数据被加载到缓存中，它们可以在训练过程中多次重复使用，而无需多次读取磁盘。这可以减少内存和 CPU 资源的使用，从而使这些资源可以用于其他任务，如模型训练和推理。

5. **提高训练效率**：缓存可以减少数据加载的等待时间，使训练过程更加高效。这对于快速迭代模型训练和实验非常有用，因为我们可以更快地看到不同设置的效果。

需要注意的是，使用缓存可能会占用一定量的内存或存储空间，因此需要根据可用的资源和数据集大小来决定是否使用缓存，以及缓存的容量。另外，缓存数据需要及时更新，以确保数据的一致性。在 YOLOv5 训练中，使用缓存通常是可选的，并可以通过相应的设置来启用或禁用。

<kbd>Summary</kbd>：简单来说，使用 cache 就是利用空间来换时间，推荐使用。

## <kbd>拓展</kbd> device

在 PyTorch 中，`model.to(1, 2, 3)` 这种写法是不正确的，因为 `.to()` 方法的参数应该是一个设备（device）对象或设备的字符串表示，而不是一系列设备索引。

正确的方式是将模型移到一个指定的设备上，通常是使用以下方式之一：

1. 移动到 CPU：
 ```python
 model.to('cpu')
 ```

2. 移动到单个 GPU（设备索引为 0）：
 ```python
 model.to('cuda:0')
 ```

3. 移动到多个 GPU（以列表形式指定多个设备）：
 ```python
 model.to(['cuda:0', 'cuda:1'])
 ```

4. 移动到当前默认的设备（通常是 GPU，如果没有 GPU 则是 CPU）：
 ```python
 model.to(torch.device('cuda' if torch. cuda.is_available() else 'cpu'))
 ```

在 YOLOv5 中同样的，先对 `args.device` / `opt.device` 进行 decoder，如果是用 CUDA，那么就变为 `['cuda:0', 'cuda:1']` 这样的形式。转换代码如下：

```python
def convert_device_str(device_str):  
    if device_str != "cpu":
        device_list = device_str.split(',')  # 将字符串按逗号分割  
        return list(map(lambda x: f'cuda:{x}', device_list))  # 将每个部分映射为 'cuda:x'  
    else:
        return "cpu"
  
args.device = convert_device_str(args.device)
print(args.device)  # ['cuda:0', 'cuda:1', 'cuda:2']
```

## <kbd>拓展</kbd> seed

在 YOLOv5 中，批量固定种子的函数如下：

```python
def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    # torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    if deterministic and check_version(torch.__version__, '1.12.0'):  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)
```

这段代码定义了一个函数 `init_seeds`，其目的是初始化随机数生成器（random number generator，RNG）的种子。这样可以确保在运行涉及随机性的代码时，每次都能得到相同的输出，这对于在训练模型时获得可重复的结果非常重要。

1. `random.seed(seed)`：使用 Python 内置的 random 库来设置随机数生成器的种子。
2. `np.random.seed(seed)`：使用 NumPy 库来设置随机数生成器的种子。
3. `torch.manual_seed(seed)`：在 PyTorch 中设置随机数生成器的种子。
4. `torch.cuda.manual_seed(seed)` 和 `torch.cuda.manual_seed_all(seed)`：这两行代码用于在 CUDA 设备上设置随机数生成器的种子。这在有多个 GPU 的情况下非常有用，它确保了在所有 GPU 上生成的随机数是确定的。
5. `torch.backends.cudnn.benchmark = True`：这是设置 CuDNN 后端的一个选项，允许在某些情况下加速卷积神经网络的操作。但是，请注意，在某些情况下，如 YOLOV5 的版本中，这个选项可能会引起问题，因此可能需要根据实际情况进行调整。
6. `if deterministic and check_version(torch.__version__, '1.12.0'):`：这段代码检查 PyTorch 的版本是否为 1.12.0 或更高。如果是，它将启用确定性算法，并设置 CuDNN 的后端为确定性模式。此外，它还设置了环境变量 `CUBLAS_WORKSPACE_CONFIG` 和 `PYTHONHASHSEED`，以确保在运行涉及随机性的代码时获得可重复的结果。

## <kbd>拓展</kbd> Single Class

在 YOLOv5 中，可以通过在运行代码时设置参数 `single_cls=True` 来启用单类别模式。在这种模式下，训练集中的所有图像都将被视为同一类别的样本，这有助于加速训练过程并提高模型的准确性。

当设置 `single_cls=True` 时，训练过程将忽略数据集中的类别信息，并将所有图像视为同一类别的样本。这使得模型能够更快地收敛，并且通常会提高模型的性能。

需要注意的是，这种设置只适用于当数据集只包含一个类别时的情况。如果数据集包含多个类别，那么设置 `single_cls=True` 可能会对模型的性能产生负面影响。

总之，通过在运行 YOLOv5 代码时设置参数 `single_cls=True`，可以启用单类别模式，这有助于加速训练过程并提高模型的准确性。

---

<kbd>By the way</kbd>：YOLOv5 官方文档中对 `single_cls` 的描述为“train multi-class data as single-class”，问过大佬，大佬的回答是“**如果开启，应该就直接不计算类别损失了**”。

## <kbd>拓展</kbd> 矩阵训练（rectangular training）
"矩形训练"是一种优化训练过程的方法，它可以减少在每个批次中的填充(padding)。在传统的训练过程中，我们通常会将所有的输入图片调整为相同的尺寸，例如 $416\times 416$。如果输入图片的原始尺寸不是正方形，那么我们需要通过填充(padding)来将其变为正方形，这个过程称之为 Letter Box。

### 1. Letter Box

LetterBox 的基本思路是：<font color='green'>先计算输入图片尺寸和输出尺寸的比例，让图片的长和宽乘上这个最小的比例，之后再进行填充（Padding）</font>。

1. **确定目标尺寸**：首先，选择一个固定的输入图像尺寸，通常是一个正方形，表示为 $(w_{\mathrm{out}}, h_{\mathrm{out}})$。

2. **计算缩放比例**：计算原始图像的宽度和高度与目标尺寸的宽度和高度之间的缩放比例（取最小比例）。这可以通过以下公式计算：
   $$
   \mathrm{scale} = \min(\frac{w_{\mathrm{out}}}{w_{\mathrm{in}}}, \frac{h_{\mathrm{out}}}{h_{\mathrm{in}}})
   $$
   
   其中，$w_{\mathrm{out}}$ 和 $h_{\mathrm{out}}$ 是目标尺寸，$w_{\mathrm{in}}$ 和 $h_{\mathrm{in}}$ 是原始图像的宽度和高度。

3. **调整图像大小**：使用缩放比例将原始图像调整为目标尺寸，保持宽高比。这可以通过以下公式计算新的图像宽度和高度：
   $$
   w_\mathrm{new} = \mathrm{round}(w_\mathrm{in} * \mathrm{scale}) \\
   h_\mathrm{new} = \mathrm{round}(h_\mathrm{in} * \mathrm{scale})
   $$

4. **计算填充**：根据目标尺寸和新调整的图像尺寸，计算需要在图像周围添加的填充。这可以通过以下公式计算：
   $$
   \begin{align}
   & \mathrm{left_{pad}} = \frac{w_\mathrm{out} - w_\mathrm{new}}{2} \\
   & \mathrm{right_{pad}} = w_\mathrm{out} - w_\mathrm{new} - \mathrm{left_{pad}} \\
   & \mathrm{top_{pad}} = \frac{h_\mathrm{out} - h_\mathrm{new}}{2} \\
   & \mathrm{bottom_{pad}} = w_\mathrm{out} - w_\mathrm{new} - \mathrm{top_{pad}}
   \end{align}
   $$

5. **添加填充**：使用上述计算得到的填充值，将调整后的图像放置在一个尺寸为 $(w_\mathrm{out}, h_\mathrm{out})$ 的画布上，其中填充区域的颜色通常是指定的填充颜色。

这样，通过 Letterbox 处理，模型可以接受具有相同固定尺寸的输入图像，而不管原始图像的尺寸如何，同时保持了图像内容的纵横比。这对于让模型处理多尺寸图像非常有用，尤其是在物体检测等任务中。

下面的 Letterbox 的代码实现：

```python
import cv2
import numpy as np


def letterbox(img: np.ndarray, out_shape=(416, 416), pad_color=(128, 128, 128)):
    if isinstance(out_shape, int):  # 如果 out_shape 是整数，将其转换为元组
        out_shape = (out_shape, out_shape)

    # 1. 确定目标尺寸
    h_out, w_out = out_shape
    
    # 获取输入图像的高度（h）、宽度（w）和通道数（在此处不使用通道数）
    h_in, w_in, _ = img.shape
    
    # 2. 计算缩放比例
    scale = min(w_out / w_in, h_out / h_in)
    
    # 3. 调整图像大小
    w_new = round(w_in * scale)
    h_new = round(h_in * scale)
    
    # 4. 计算填充
    left_pad = (w_out - w_new) // 2
    right_pad = w_out - w_new - left_pad
    top_pad = (h_out - h_new) // 2
    bottom_pad = h_out - h_new - top_pad
    
    # 5. 添加 Padding
    img = cv2.copyMakeBorder(cv2.resize(img, (w_new, h_new)), top_pad, bottom_pad, left_pad, right_pad, 
                             cv2.BORDER_CONSTANT, value=pad_color)

    # 返回调整后的图像
    return img


img = cv2.imread("/mnt/f/Projects/本地代码/yolov5/Le0v1n/Tom-and-jerry.jpg")

img_letterbox = letterbox(img, out_shape=(416, 416))

# 输出长度和宽度
print(f"原始图像: 图像宽度×长度: {img.shape[0]}×{img.shape[1]}")
print(f"letterbox: 图像宽度×长度: {img_letterbox.shape[0]}×{img_letterbox.shape[1]}")

cv2.imwrite('Image.jpg', img_letterbox)
```

```
原始图像: 图像宽度×长度: 287×356
letterbox: 图像宽度×长度: 416×416
```

<div align=center>
    <img src=./imgs_markdown/原图v.s.LetterBox.png
    width=100%>
</div>

### 2. Rectangular

在矩形训练中，我们会保持图片的原始长宽比，只对较短的边进行缩放和最小程度的填充，以满足模型的输入要求。这样可以减少冗余信息，提高模型的训练效率和性能。这种方法在处理有大量非正方形图片的数据集时特别有效。

<font color='blue'>矩形训练也很好理解，也就是先将较长边设定为目标尺寸 416，短边按比例缩放，再对短边进行少量的填充（padding）使短边满足 32 的倍数</font>。

```python
import cv2
import numpy as np


def letterbox(img: np.ndarray, out_shape=(416, 416), pad_color=(128, 128, 128)):
    if isinstance(out_shape, int):  # 如果 out_shape 是整数，将其转换为元组
        out_shape = (out_shape, out_shape)

    # 获取输入图像的高度（h）、宽度（w）和通道数（在此处不使用通道数）
    h, w, _ = img.shape

    # 计算高度和宽度的调整比例以保持纵横比
    if h > w:  # 如果图像的高度大于宽度
        r = out_shape[1] / h  # 计算高度的缩放比例以保持纵横比
        h_ = out_shape[1]  # 使用指定的输出高度
        w_ = int(round(w * r))  # 根据缩放比例计算宽度
    else:  # 如果图像的宽度大于高度
        r = out_shape[0] / w  # 计算宽度的缩放比例以保持纵横比
        w_ = out_shape[0]  # 使用指定的输出宽度
        h_ = int(round(h * r))  # 根据缩放比例计算高度

    # 调整图像大小
    img = cv2.resize(img, (w_, h_))

    left_pad = (out_shape[0] - w_) // 2  # 计算左填充的宽度，使图像水平居中
    right_pad = out_shape[0] - w_ - left_pad  # 计算右填充的宽度，以确保图像总宽度等于指定的输出宽度
    top_pad = (out_shape[1] - h_) // 2  # 计算上填充的高度，使图像垂直居中
    bottom_pad = out_shape[1] - h_ - top_pad  # 计算下填充的高度，以确保图像总高度等于指定的输出高度

    # 使用cv2.copyMakeBorder函数添加填充
    img = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=pad_color)

    # 返回调整后的图像
    return img


def rectangular(img: np.ndarray, out_shape=416, pad_color=(128, 128, 128)):
    if isinstance(out_shape, (tuple, list)):  # 如果out_shape是元组或列表，将其转换为单个值
        out_shape = out_shape[0]
    
    # 获取输入图像的高度（h）、宽度（w）和通道数（在此处不使用通道数）
    h, w, _ = img.shape

    # 根据图像的纵横比，计算调整后的高度和宽度
    if h > w:
        r = out_shape / h
        h_ = out_shape
        w_ = int(round(w * r))
    else:
        r = out_shape / w
        h_ = int(round(h * r))
        w_ = out_shape

    # 调整图像大小，确保纵横比不变
    img = cv2.resize(img, (w_, h_))

    # 初始化左、右、上、下填充的宽度
    left_pad, right_pad, top_pad, bottom_pad = 0, 0, 0, 0

    # 如果图像宽度不是32的倍数，计算左右填充
    if w_ % 32 != 0:
        left_pad = (32 - (w_ % 32)) // 2
        right_pad = 32 - (w_ % 32) - left_pad

    # 如果图像高度不是32的倍数，计算上下填充
    if h_ % 32 != 0:
        top_pad = (32 - (h_ % 32)) // 2
        bottom_pad = 32 - (h_ % 32) - top_pad

    # 使用cv2.copyMakeBorder函数添加填充，以确保图像的最终形状是32的倍数
    img = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=pad_color)

    # 返回调整后的图像
    return img


if __name__ == "__main__":
    # 读取图片
    img = cv2.imread("/mnt/f/Projects/本地代码/yolov5/Le0v1n/Tom-and-jerry.jpg")

    # 进行 letterbox 和 rectangular 处理
    img_letterbox = letterbox(img, out_shape=(416, 416))
    img_rectangular = rectangular(img, out_shape=(416, 416))

    # 输出长度和宽度
    print(f"原始图像: 图像宽度×长度: {img.shape[0]}×{img.shape[1]}")
    print(f"letterbox: 图像宽度×长度: {img_letterbox.shape[0]}×{img_letterbox.shape[1]}")
    print(f"rectangular: 图像宽度×长度: {img_rectangular.shape[0]}×{img_rectangular.shape[1]}")

    cv2.imwrite('letterbox.jpg', img_letterbox)
    cv2.imwrite('rectangular.jpg', img_rectangular)
    cv2.imwrite('combined_image.jpg', img_rectangular)
```

```
原始图像: 图像宽度×长度: 287×356
letterbox: 图像宽度×长度: 416×416
rectangular: 图像宽度×长度: 352×416
```

<div align=center>
    <img src=./imgs_markdown/原图v.s.LetterBoxv.s.Rectangular.png
    width=100%>
</div>

<kbd>Note</kbd>: 但是这样做了会引入新的问题 —— 数据集中每个 Batch 中图片的大小不一样，YOLO 的处理是将尺寸接近的放到一起处理，这就导致不能使用 dataloader 中的 `shuffle` 功能。

## <kbd>拓展</kbd> 损失函数权重

### 1. box 损失权重

### 2. cls 损失权重

### 3. dfl（Distribution Focal Loss）损失权重

在 YOLOv5 中，dfl 是一种新的损失函数，其全称为 Distribution Focal Loss。它是在 Focal Loss 的基础上进行了改进，主要解决了类别不平衡问题。在传统的 Focal Loss 中，对于难以分类的样本，损失函数会放大其权重，提升其在训练中的重要性。而在 dfl 中，损失函数不仅考虑了样本的难易程度，还考虑了样本的分布情况。具体来说，dfl 会根据样本的概率分布来计算损失，使得那些分布较为集中的样本（即概率较高或较低的样本）的损失减小，而那些分布较为均匀的样本（即概率接近 0.5 的样本）的损失增大。这样可以使得模型更关注那些不确定性较高的样本，从而提高模型的泛化能力。dfl 损失函数的公式如下：

$$
\text {DFL} \left (p_ {t}\right)=-\alpha_ {t}\left (1-p_ {t}\right)^ {\gamma} \log \left (p_ {t}\right) \cdot \left (\frac {p_ {t}} {\bar {p}}\right)^ {\beta}
$$

其中，$p_t$ 是样本的真实概率，$\alpha_t$ 和 $\gamma$ 是 Focal Loss 中的参数，$\bar{p}$ 是所有样本概率的均值，$\beta$ 是一个超参数，用于控制分布因子的影响程度。当 $\beta=0$ 时，dfl 退化为Focal Loss。当 $\beta>0$ 时，dfl 会增加分布较为均匀的样本的损失；当 $\beta<0$ 时，dfl 会增加分布较为集中的样本的损失。



# 知识来源

1. [Ultralytics YOLOv5 Docs](https://docs.ultralytics.com/yolov5/)
2. [rectangular training 矩阵训练](https://blog.csdn.net/REstrat/article/details/126851437)