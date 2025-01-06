# YOLO

## 数据增强

### 标签格式

```python
labels = {
    "img": image,  # 输入图像
    "cls": np.array([0, 1]),  # 类别标签
    "instances": Instances(...),  # 边界框、分割和关键点的标签对象实例
    "mosaic_border": Tuple[int, int]  # [可选项] 马赛克增强的边框大小
}
```

经过数据增强后：

```python
labels = {
    "img": np.ndarray,  # 变换后的图像
    "cls": np.ndarray,  # 更新后的类别标签
    "instances": Instances,  # 更新后的对象实例（边界框、分割和关键点的标签对象实例）
    "resized_shape": Tuple[int, int]  # 变换后的新图像形状
}
```

标签常用3种坐标表示方式：

- `xyxy` means left top and right bottom
- `xywh` means center x, center y and width, height(YOLO format)
- `ltwh` means left top and width, height(COCO format)

### num_workers

- 如果是train模式，`workers`没有变化
- 如果是val模式，`workers = workers * 2`

### Rect

- 当`rect=True`时：
  - 与马赛克不兼容
  - 与Mixup不兼容
  - 与`shuffle=True`不兼容

### Mosaic

- 概率必须在$[0, 1]$之间
- 马赛克的grid size必须是4或9
  - 4=2*2
  - 9=3*3

### Mixup

...

### CopyPaste

CopyPaste 类，用于对图像数据集应用复制粘贴增强。此类实现了论文《[Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation](https://arxiv.org/abs/2012.07177)》中描述的复制粘贴增强技术。它可以结合来自不同图像的对象以创建新的训练样本。

> ⚠️ 使用此增强方式的数据集必须要有分割的标签，否则不会执行
> ⚠️ CopyPaste有两种方式：
> 1. flip
> 2. mixup

## dataloader

- 只使用主线程构建dataloader对象
- 🚨在进行`val`时，`rect=True`


## 自动计算batch

- `batch=-1`：自动计算60%GPU显存占用的batch size
- `batch=0.8`：自动计算80%GPU显存占用的batch size

> 🚨上面两种情况只适用于单卡的情况，对于多卡的情况Ultralytics代码暂不支持！
> 🚨如果开启了`torch.backends.cudnn.benchmark=True`，那么也是不支持的！
> 🚨自动计算的batch范围为：$[1, 1024]$，超出则使用默认的16作为batch

该功能的核心函数为：`ultralytics/utils/autobatch.py/autobatch()`

## best.pt计算方式

在 `ultralytics/utils/metrics.py` 文件中的 `Metric` 类的 `fitness()` 方法中：

```python
def fitness(self):
    """Model fitness as a weighted combination of metrics.
        将模型拟合度视为各项指标的加权组合。
    """
    # 权重分别对应 [Precision, Recall, mAP@0.5, mAP@0.5:0.95]
    # 默认权重 [0.0, 0.0, 0.1, 0.9]，可以根据需求调整
    w = [0.25, 0.25, 0.25, 0.25]  # 修改后的权重
    return (np.array(self.mean_results()) * w).sum()
```

默认情况下，`fitness score` 的计算公式为：

$$
\text{fitness score} = 0 \times \text{Precision} + 0 \times \text{Recall} + 0.1 \times \text{mAP@0.5} + 0.9 \times \text{mAP@0.5:0.95}
$$

其中：
- **`mAP@0.5`**：衡量 IoU ≥ 0.5 时的平均精度，代表较宽松条件下的模型性能。
- **`mAP@0.5:0.95`**：综合了多个 IoU 阈值（从 0.5 到 0.95，步长为 0.05），提供了更严格和全面的性能评估。

在训练和验证阶段，IoU 阈值的主要作用包括：
1. **计算定位损失**：衡量预测框与真实框的重叠程度，优化模型的定位能力。
2. **划分正负样本**：根据 IoU 阈值，将预测框划分为正样本（IoU > 0.5）、忽略样本（0.3 ≤ IoU ≤ 0.5）和负样本（IoU < 0.3），从而影响分类损失和定位损失的计算。

Ultralytics 默认权重设置为 10% 的 `mAP@0.5` 和 90% 的 `mAP@0.5:0.95`，分别对应 VOC 和 COCO 数据集的评估指标。但在实际业务场景中，可以根据需求调整权重：

- 如果关注 **漏检**（Recall），应适当增加 Recall 的权重。
- 如果关注 **误检**（Precision），应适当增加 Precision 的权重。

## imgsz

- `train`和`val`模式的`imgsz`必须是一个整数，如`640`，之后程序会将其转换为`[640, 640]`
- `predict`和`export`模式的`imgsz`可以是一个整数，如`640`，也可以是一个列表，如`[640, 480]`

```bash
yolo train imgsz=640        # ✅
yolo train imgsz=640,480    # ❌

yolo val imgsz=640          # ✅
yolo val imgsz=640,480      # ❌

yolo predict imgsz=640      # ✅
yolo predict imgsz=640,480  # ✅

yolo export imgsz=640       # ✅
yolo export imgsz=640,480   # ✅
```

## grid cell

grid cell的最小值为32，依据如下：

`ultralytics/engine/trainer.py/BaseTrainer/_setup_train`

```python
# Check imgsz（grid cell的大小最小是32*32）
gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
```

## AMP

1. 在`.train()`方法中开启了`amp=True`（默认开启）
2. 在主线程中先备份回调函数防止在测试AMP时对其改动
3. 使用`check_amp(model)`的函数来进行测试，该函数返回一个bool
   1. 先将图片（`ultralytics/assets/bus.jpg`）复制8遍形成一个batch
   2. 因为是验证是否可用，因此imgsz的最大值被限制为256
   3. 先试用FP32的模型对结果进行推理，得到结果`a`
   4. 之后使用`with autocast(enable=True)`上下文管理器中再次使用模型进行推理，得到结果`b`
   5. 先判断`a`和`b`的shape是否相等
   6. 之后再使用`torch.allclose(a, b.float(), atol=0.5)`判断二者的绝对容忍度是否在0.5的范围内
   7. `True` -> 可以开启AMP
   8. `False` -> 不开启AMP（根据异常原因进行不同的告警）
      1. `except ConnectionError`：网络连接异常，直接跳过检查了（返回值依然是False）
      2. `except (AttributeError, ModuleNotFoundError)`：原项目被修改导致模型无法被正常加载
      3. `except AssertionError`：未知的原因导致`a`和`b`的shape不相等或二者的绝对容忍度不在0.5的范围内

> 🔔不管AMP是否为True都会创建`torch.amp.GradScaler`，只不过`enabled=self.amp`

## 初始化策略

```python
def initialize_weights(model):
    """Initialize model weights to random values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True
```

> Issue链接：[Initialization of yolov8 #12677](https://github.com/ultralytics/ultralytics/issues/12677)

### Conv2d

glenn-jocher说是不使用kaiming初始化了，因此直接设置为`pass`，但是根据PyTorch官方代码，`nn.Conv2d`在创建时自动就应用了kaiming初始化方式，代码如下：

```python
class _ConvNd(Module):
	...
	
    def __init__(...) -> None:
        super(_ConvNd, self).__init__()
		
		...
		
        self.reset_parameters()  # 🚨 Here!

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # 🚨 Here!
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
```

### BatchNorm2d

Ultralytics团队将其设置为：

- `eps = 1e-3`
- `momentum = 0.03`

根据glenn-jocher的说法，这是一个经验值。

# Coding

## PyTorch

### torch.Generate()

`torch.Generator` 是 PyTorch 中用于**生成随机数**的类。它负责管理生成伪随机数的算法状态，并在许多就地（in_place）随机抽样函数中作为关键字参数使用。这个类的主要功能包括：

1. **设备指定**：`torch.Generator` 可以被指定在特定的设备上创建，如 CPU 或 GPU，这通过构造函数中的 `device` 参数实现。

2. **种子设置**：可以通过 `manual_seed(seed)` 方法设置用于生成随机数的种子，确保结果的可重复性。

3. **状态管理**：`get_state()` 方法以 `torch.ByteTensor` 的形式返回生成器的状态，而 `set_state(new_state)` 方法则用于设置生成器的状态。

4. **初始种子获取**：`initial_seed()` 方法返回用于生成随机数的初始种子。

5. **随机数生成**：`seed()` 方法从 `std::random_device` 或当前时间获取一个非确定性的随机数，并用它来设置生成器的种子。

`torch.Generator` 提供了可预测性和可复现性，这对于调试和复现实验结果非常有用。同时，它是线程安全的，支持在多线程环境中使用，并且支持多种随机数分布。在实际应用中，`torch.Generator` 广泛用于机器学习和深度学习中的随机数生成需求，如初始化模型权重、随机打乱训练数据等。

### with torch_distributed_zero_first(rank)

`with torch_distributed_zero_first(rank)` 是 PyTorch 分布式训练中的一个上下文管理器，用于确保在分布式训练中，只有 rank（即进程的序号）为 0 的进程（主进程）首先执行某些操作，而其他进程则等待主进程完成这些操作后再继续执行。这个上下文管理器的作用是同步不同进程间的操作，以避免数据不一致或者资源竞争的问题。

具体来说，`torch_distributed_zero_first` 函数的工作原理如下：

1. 如果当前进程的 `local_rank` 不是 `-1` 或 `0`（即不是主进程），那么它将调用 `torch.distributed.barrier()` 函数，这会导致进程阻塞，直到所有进程都到达这个 barrier。

2. 然后，上下文管理器会执行 `yield`，这意味着它会暂停执行，允许 `with` 代码块内的代码运行。只有当 `local_rank` 为 `0` 时，这部分代码才会执行。
   > 🚨`yield`会抛出后面跟着的值（如果有）并且等待下次next

3. 在 `with` 代码块执行完毕后，如果 `local_rank` 是 `0`（即主进程），那么会再次调用 `torch.distributed.barrier()`，这样所有等待在第一个 barrier 的进程现在可以继续执行。

这种机制确保了在分布式训练中，一些只需要在一个进程中执行一次的操作（比如加载数据集、下载预训练模型等）不会在每个进程中重复执行，同时也保证了这些操作的结果能够被其他进程所使用。通过这种方式，可以提高分布式训练的效率和一致性。

### torch.cuda.empty_cache()

`torch.cuda.empty_cache()` 是 PyTorch 提供的一个函数，用于释放当前 PyTorch CUDA 分配器中的未被引用的缓存内存。这个函数的作用包括：

1. **释放未占用内存**：它会释放那些不再被任何张量引用的内存块，这些内存块可能由于之前的操作而被分配但未被释放。

2. **减少内存碎片**：在进行大量的内存分配和释放操作时，可能会产生内存碎片。调用 `empty_cache()` 可以帮助减少这种碎片，从而提高内存使用效率。

3. **提高内存可用性**：释放的内存可以被操作系统回收，或者被其他 CUDA 应用程序使用。这有助于在多 GPU 应用或者需要大量内存的应用中保持内存的可用性。

4. **影响性能**：虽然 `empty_cache()` 可以释放内存，但是频繁调用它可能会导致性能下降，因为它会打断 PyTorch 的内存分配策略，使得后续的内存分配可能需要更多的时间来找到合适的内存块。

5. **nvidia-smi 可见性**：释放的内存将变得对 `nvidia-smi` 可见，这意味着你可以在使用 `nvidia-smi` 命令时看到更多的可用 GPU 内存。

需要注意的是，`empty_cache()` 并不释放那些仍然被张量引用的内存。如果张量仍然存在并且被引用，那么它们所占用的内存不会被释放。此外，`empty_cache()` 并不影响 CPU 内存，它只影响 GPU 内存。

在实际使用中，通常不需要手动调用 `empty_cache()`，因为 PyTorch 的内存管理器会自动处理内存的分配和释放。但在某些特定的应用场景下，比如在长时间的训练过程中，或者在内存非常紧张的情况下，合理地使用 `empty_cache()` 可以帮助优化内存使用。


### torch.beckends.cudnn.benchmark

`torch.backends.cudnn.benchmark=True` 是 PyTorch 中的一个设置，用于优化卷积神经网络（CNN）的性能。具体来说，这个设置的作用如下：

1. **自动调优**：当设置为 `True` 时，PyTorch 会使用 cuDNN 库（NVIDIA 提供的一个用于深度学习的 GPU 加速库）来自动寻找最佳的卷积算法。这意味着在执行卷积操作之前，cuDNN 会尝试不同的卷积算法，并选择在当前硬件和输入尺寸下最快的那个。

2. **性能提升**：这可以显著提高卷积神经网络的性能，特别是在使用 GPU 进行训练或推理时。自动选择最优算法可以减少计算时间，提高吞吐量。

3. **额外的计算开销**：需要注意的是，自动调优过程本身会有一定的计算开销，因为它需要尝试不同的算法。
   > ⚠️如果你的模型或数据集非常小，或者你的训练循环迭代非常快，这种额外的开销可能会抵消性能提升，甚至导致总体性能下降。

4. **适用场景**：这个设置在模型结构固定，且输入尺寸变化不大的情况下效果最好，因为这样 cuDNN 可以更好地预测和选择最优算法。

5. **默认值**：在 PyTorch 中，`torch.backends.cudnn.benchmark` 的默认值是 `False`，这意味着不会自动进行算法选择，而是使用默认的算法。

在使用这个设置时，你应该根据你的具体情况（模型大小、输入尺寸、训练速度等）来决定是否启用它。如果你发现启用后性能没有提升，或者出现了其他问题，可以尝试将其设置回 `False`。

### torch.cuda.device_count()

`torch.cuda.device_count()` 是 PyTorch 中的一个函数，用于返回当前系统中可用的 CUDA 设备（即 NVIDIA GPU）的数量。这个函数可以帮助你确定你的机器上有多少个 GPU 可以用于加速计算。

当你在一个装有多个 GPU 的机器上运行 PyTorch 程序时，`torch.cuda.device_count()` 会返回这些 GPU 的总数。这允许你的程序动态地适应不同的硬件配置，例如，你可以根据可用的 GPU 数量来决定是否使用数据并行处理。

下面是如何使用这个函数的一个简单示例：

```python
import torch

# 检查是否有可用的 CUDA 设备
if torch.cuda.is_available():
    # 获取 CUDA 设备的数量
    num_gpus = torch.cuda.device_count()
    print(f'Number of CUDA devices available: {num_gpus}')
else:
    print('No CUDA devices are available.')
```

这段代码首先检查 CUDA 是否可用（即是否有支持 CUDA 的 GPU 和正确安装的驱动），然后使用 `torch.cuda.device_count()` 来获取并打印可用的 GPU 数量。如果没有可用的 CUDA 设备，它会打印相应的消息。

🚨==计算得到的GPU设备会因为环境变量而改变==，如下所示：

```python
import torch

print(f"{torch.cuda.device_count() = }")
```

```bash
# 不添加环境变量，得到结果是8
python temp.py

# 添加环境变量后，得到的结果是2
CUDA_VISIBLE_DEVICES=0,1 python temp.py
```

## Python

### warnings.filterwarnings()

`warnings.filterwarnings()` 是 Python 标准库 `warnings` 模块中的一个函数，用于控制哪些类别的警告应该被显示，哪些应该被忽略。这个函数允许开发者在运行时动态地控制警告信息的过滤，而不是在代码中静态地定义。

### 函数原型

```python
warnings.filterwarnings(
    action,  # 如何处理警告
    category=None,  # 指定警告的类别
    message='',  # 指定警告消息的字符串（🚨需要匹配后前面的action才会生效）
    module='',  # 指定模块名的字符串
    lineno=0,  # 指定行号的整数
    append=False  # 是否将过滤规则追加到当前的警告过滤列表中
)
```

### 参数说明

- `action`：指定如何处理警告。常用的值包括：
  - `'ignore'`：忽略警告，不显示也不记录。
  - `'default'`：使用默认行为（显示警告）。
  - `'always'`：总是显示警告。
  - `'module'`：显示警告，但只有当警告发生在触发者的模块级别时。
  - `'once'`：只显示一次警告，之后忽略。
  - `'error'`：将警告当作错误处理。
- `category`：指定警告的类别，默认为 `Warning`。可以指定为：
	- `Warning`：这是最基本的警告类别，用于指示一般性的警告信息，没有特定的类别时会使用这个。
	- `DeprecationWarning`：当使用的功能已经被官方标记为弃用，未来版本中可能会移除时，会触发此类警告。
	- `PendingDeprecationWarning`：与 DeprecationWarning 类似，但是用于那些即将在未来版本中被弃用的功能。
	- `SyntaxWarning`：当 Python 代码中存在可能的语法问题时，会触发此类警告。例如，当使用了一个保留字作为变量名，或者代码中存在语法上不推荐的做法时。
	- `RuntimeWarning`：这类警告通常与程序运行时可能发生的问题相关，比如除以零、无效的数组索引等。
	- `FutureWarning`：当代码使用了在未来版本中可能改变或不再支持的特性时，会触发此类警告。
	- `ImportWarning`：当模块导入过程中存在潜在问题时，会触发此类警告，比如重复导入模块或者导入了不存在的模块。
	- `UnicodeWarning`：当处理 Unicode 字符串时存在潜在问题，比如编码或解码问题，或者在比较不同编码的字符串时，会触发此类警告。
	- ...

- `message`：指定警告消息的字符串。如果提供了此参数，只有当警告消息匹配时，`action` 才会被应用。
- `module`：指定模块名的字符串。如果提供了此参数，只有当警告来自指定模块时，`action` 才会被应用。
- `lineno`：指定行号的整数。如果提供了此参数，只有当警告来自指定行号时，`action` 才会被应用。
- `append`：布尔值，指定是否将过滤规则追加到当前的警告过滤列表中。默认为 `False`，即替换当前的过滤规则。

### 使用示例

```python
import warnings

# 忽略所有 DeprecationWarning 警告
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 将特定消息的警告当作错误处理
warnings.filterwarnings('error', message='invalid')

# 只显示一次特定模块的特定类别警告
warnings.filterwarnings('once', category=RuntimeWarning, module='mymodule')
```

使用 `warnings.filterwarnings()` 可以帮助开发者在开发过程中管理警告信息，避免被过多的警告干扰，同时也能够在需要时捕捉到重要的警告信息。

### \# noqa

在Python代码中，`# noqa` 是一个注释标记，用于告诉代码静态分析工具或者linter（例如Pylint、Flake8等）忽略当前行或特定行的警告或错误。这个标记通常用在代码中那些故意违反了某些编码规则，但又不希望因此产生linter警告的地方。

例如，假设有一个代码片段违反了某个linter的规则，但你确定这是正确的做法，或者暂时不想处理这个警告，你可以在该行代码后面添加`# noqa`来忽略警告：

```python
# 这是一个可能违反linter规则的代码行
some_variable = 10  # noqa: F401
```

在这个例子中，`# noqa: F401` 告诉linter忽略这一行的“未引用的导入”（F401）警告。

`# noqa` 可以单独使用，也可以指定忽略特定的警告代码：

- `# noqa`：忽略当前行的所有警告。
- `# noqa: <代码>`：忽略特定代码的警告，其中 `<代码>` 是linter的警告代码。

使用`# noqa`是一种快速抑制警告的方法，但它应该谨慎使用，因为它可能会掩盖潜在的问题。最佳实践是尽量修正代码以符合linter的规则，而不是简单地忽略警告。

### def \_\_setitem\_\_()

`__setitem__` 方法是 Python 中的一个特殊方法（也称为魔术方法），它被用来实现对象的项赋值功能。当你使用类似 `obj[1] = xxx` 这样的语法对对象的某个项进行赋值时，实际上会调用该对象的 `__setitem__` 方法。

具体来说，`__setitem__` 方法接受两个参数：第一个参数是用于指定项位置的索引（可以是整数、切片或者任何可哈希的对象），第二个参数是你要赋给该项的值。

下面是 `__setitem__` 方法的一个简单示例：

```python
class MyList:
    def __init__(self):
        self.data = []
        
    def __setitem__(self, index, value):
        self.data[index] = value

# 创建 MyList 的实例
my_list = MyList()

# 使用 __setitem__ 方法赋值
my_list[0] = 'Hello'  # 这会调用 MyList 的 __setitem__ 方法
print(my_list.data)  # 输出: ['Hello']
```

在这个例子中，`MyList` 类有一个 `data` 属性，它是一个普通的 Python 列表。`__setitem__` 方法被定义为将值赋给 `data` 列表在 `index` 索引处的位置。当我们对 `my_list` 实例使用 `my_list[0] = 'Hello'` 这样的赋值语句时，实际上是在调用 `MyList` 类的 `__setitem__` 方法。

### random.choice(seq)

`random.choice()` 方法是 Python 标准库中 `random` 模块提供的一个函数，它用于从给定的序列（如列表、元组等）中随机选择一个元素并返回。

### random.uniform(a, b)

`random.uniform(a, b)` 函数是 Python 中 random 模块提供的一个函数，它用于生成一个指定范围内的随机浮点数。

> 🔔范围是闭区间，包含`a`和`b`

### os.cpu_count()

`os.cpu_count()` 是 Python 标准库 `os` 模块中的一个函数，用于返回当前机器上可用的 CPU 核心数。这个函数返回的值包括物理核心和逻辑核心（如果操作系统支持超线程技术，如 Intel 的 Hyper-Threading）。逻辑核心是现代多核处理器上的一个特性，它们允许单个物理核心同时处理多个线程。

这个函数可以帮助你了解机器的并发处理能力，从而在需要时进行资源分配和并行计算的优化。例如，当你需要决定如何分配任务到多个线程或者进程时，可以使用 `os.cpu_count()` 来获取机器的 CPU 核心数。

以下是如何使用 `os.cpu_count()` 的一个简单示例：

```python
import os

# 获取当前机器的 CPU 核心数
cpu_cores = os.cpu_count()
print(f'Number of CPU cores available: {cpu_cores}')
```

这段代码会打印出当前机器上可用的 CPU 核心数。如果 `os.cpu_count()` 返回 `None`，则表示无法确定 CPU 核心数。这种情况比较少见，通常发生在某些非本地操作系统环境或者特定的虚拟化环境中。

---

我的CPU信息为：

```
Intel(R) Core(TM) i7-14700HX

基准速度:	2.10 GHz
插槽:	1
内核:	20
逻辑处理器:	28
虚拟化:	已启用
L1 缓存:	1.8 MB
L2 缓存:	28.0 MB
L3 缓存:	33.0 MB

利用率	10%
速度	2.58 GHz
正常运行时间	0:08:54:57
进程	401
线程	8664
句柄	197712
```

运行之后得到的结果为：

```
os.cpu_count() = 28
```

### yield

在编程中，`yield` 是一个关键字，它用于定义一个生成器（generator）函数。生成器是一种特殊类型的迭代器，它允许你逐个产生值，而不是一次性计算并返回所有值。这使得生成器在处理大量数据时非常有用，因为它们可以帮助节省内存。

以下是 `yield` 的一些关键点：

1. **生成器函数**：包含 `yield` 的函数被称为生成器函数。当函数执行到 `yield` 语句时，它会生成一个值并暂停执行，保留当前函数的状态，包括所有变量的值和调用堆栈。

2. **逐个产生值**：每次对生成器函数的下一次迭代请求时，函数会从上次 `yield` 暂停的地方继续执行，直到遇到下一个 `yield` 或者函数结束。

3. **惰性计算**：`yield` 允许惰性计算，即只有在需要时才计算下一个值，这可以提高效率，尤其是在处理大型数据集时。

4. **控制流**：`yield` 可以用来控制函数的执行流程，允许函数在不同的点暂停和恢复。

5. **与 `return` 的区别**：`return` 关键字用于结束函数的执行并返回一个值，而 `yield` 用于生成值但并不结束函数的执行。

下面是一个简单的 Python 示例，展示了如何使用 `yield`：

```python
def simple_generator():
    print("Start of function")
    yield 1  # 🚨抛出后等待下次next
    yield 2  # 🚨抛出后等待下次next
    print("End of function")

# 创建生成器对象
gen = simple_generator()

# 获取生成器的值
print(next(gen))  # 输出: Start of function, 然后输出: 1
print(next(gen))  # 输出: 2
print(next(gen))  # 输出: End of function, 然后引发 StopIteration 异常
```

在这个例子中，`simple_generator` 函数是一个生成器，它在每次迭代时产生一个值。当所有 `yield` 语句都被执行完毕后，函数执行结束，并抛出 `StopIteration` 异常，表示生成器已经没有更多的值可以产生了。

再举一个例子：

```python
def countdown(n):
    print("Counting down from", n)
    while n > 0:
        yield n
        n -= 1
    print("Countdown complete")

# 创建生成器对象
count_gen = countdown(5)

# 迭代生成器对象
for number in count_gen:  # 🚨因为是直接遍历的，所以不会越界，也就不会抛出StopIteration异常
    print(number)
    time.sleep(1)
```

在这个例子中，`countdown` 函数是一个生成器，它从参数 `n` 开始倒数，每次迭代产生一个递减的值，直到 `n` 减到 0。每次调用 `next(count_gen)` 或者在 `for` 循环中迭代 `count_gen` 时，生成器都会执行到下一个 `yield` 语句，并产生下一个值。

输出结果将是：

```
Counting down from 5
5
4
3
2
1
Countdown complete
```

每次迭代都会打印当前的倒数值，直到倒数结束，然后打印 "Countdown complete"。这个生成器函数可以用于任何需要逐个处理序列值的场景，例如文件逐行读取、数据处理流水线等。

### \_\_name\_\_

在Python中，`.__name__` 是一个特殊的属性，用于获取一个模块、类、函数或方法的名称。这个属性是内置的，不需要你手动设置。

- 对于模块来说，`__name__` 包含了模块的名字。如果模块是直接运行的，那么 `__name__` 将会是 `'__main__'`。
- 对于类来说，`__name__` 包含了类的名字。
- 对于函数和方法来说，`__name__` 包含了函数或方法的名字。

这里有几个例子来说明 `.__name__` 的使用：

1. **模块名称**：
```python
# 假设有一个名为 mymodule.py 的文件
print(__name__)  # 如果直接运行 mymodule.py，输出将会是 '__main__'
```

2. **类名称**：
```python
class MyClass:
    pass

print(MyClass.__name__)  # 输出：MyClass
```

3. **函数名称**：
```python
def my_function():
    pass

print(my_function.__name__)  # 输出：my_function
```

4. **数据类型**

```python
a = 'Hello'

print(f"{type(a) = }")  # <class 'str'>
print(f"{type(a).__name__ = }")  # 'str'
```

`.__name__` 属性在很多情况下都很有用，比如在创建钩子（hooks）或者插件（plugins）时，你可能会根据函数或类的名称来执行不同的操作。此外，它也是模块导入机制中的一个重要部分，特别是在包（packages）和子模块（submodules）中。

## OpenCV

### 仿射变换和透视变换

`cv2.warpPerspective` 和 `cv2.warpAffine` 都是 OpenCV 库中用于图像变换的函数，但它们在变换的复杂性和应用场景上有所不同。

- `cv2.warpPerspective`
  - **透视变换**：`cv2.warpPerspective` 实现的是透视变换，这是一种更复杂的变换，可以模拟相机视角的变化，包括旋转、缩放、平移以及视角扭曲。
  - **变换矩阵**：它需要一个3x3的变换矩阵，这个矩阵可以包含旋转、缩放、平移和透视效果。
  - **应用场景**：适用于需要模拟相机视角变化的场景，比如3D效果的模拟、图像校正等。

- `cv2.warpAffine`
  - **仿射变换**：`cv2.warpAffine` 实现的是仿射变换，这是一种线性变换，可以包含旋转、缩放、平移和剪切。
  - **变换矩阵**：它需要一个2x3的变换矩阵，这个矩阵只能包含旋转、缩放、平移和剪切效果，不能包含透视效果。
  - **应用场景**：适用于大多数需要线性变换的场景，比如图像的旋转、缩放、平移等。

- **联系**
  - **基础变换**：仿射变换可以看作是透视变换的一个特例，当透视变换矩阵的第三行是[0, 0, 1]时，透视变换就退化成了仿射变换。
  - **参数**：两者都需要指定输出图像的大小（`dsize`），并且都可以指定边界外像素的值（`borderValue`）。
  - **用途**：两者都广泛用于图像处理和计算机视觉中，用于图像的预处理、增强和特征提取。

- **区别**
  - **变换能力**：透视变换比仿射变换能模拟更多的视角效果，因为它包含了仿射变换的所有能力，并且还包括了视角扭曲的效果。
  - **性能**：由于透视变换的计算复杂度高于仿射变换，所以在处理速度上，仿射变换通常更快。
  - **参数数量**：透视变换需要更多的参数（一个3x3矩阵），而仿射变换只需要一个2x3矩阵。

在选择使用哪个函数时，需要根据具体的应用场景和所需的变换效果来决定。如果需要简单的旋转、缩放或平移，可以使用`cv2.warpAffine`；如果需要更复杂的视角变化，比如模拟相机视角或图像校正，则使用`cv2.warpPerspective`。

## numpy

### ndarray.view(1, -1)和ndarray.shape

`ndarray.view(1, -1)` 实际上是一个 NumPy 中的操作，通常用于调整数组的形状或创建新的视图（view）。不过，这里需要明确的是，`view` 和 `reshape` 是两种不同的操作，可能有一定的混淆。

在 NumPy 中，`view` 是一个用于创建数组的**新视图**（即共享相同数据内存）的方法，`reshape` 是用于改变数组形状的。而 `view` 的语法并不直接支持 `(1, -1)` 的形式。你可能混淆了 `view()` 和 `reshape()` 的用法。

接下来分开解释这两者的用法：

#### 1. ndarray.view()
`view` 方法是用来创建一个**新视图**，这个视图共享原数组的内存，但可以改变数据的解释方式。它通常用来改变数据类型。

**语法**:
```python
ndarray.view(dtype=None)
```

- `dtype`：指定新的数据类型，改变数组的解释方式。
- 返回值：一个新的数组视图，它与原数组共享相同的数据内存。

**示例**：

```python
import numpy as np

# 创建一个数组
arr = np.array([1, 2, 3, 4], dtype=np.int32)

# 创建一个新视图，修改数据类型为 uint8
new_view = arr.view(np.uint8)

print("原始数组：", arr)
print("视图数组：", new_view)
```

**输出**：
```
原始数组： [1 2 3 4]
视图数组： [1 0 0 0 2 0 0 0 3 0 0 0 4 0 0 0]
```

在这个例子中，`arr` 是以 `int32` 类型存储，每个数字占 4 字节。而 `new_view` 将同样的内存解释为 `uint8`，因此每个字节被单独提取出来。

#### 2. ndarray.reshape()

如果你想改变数组的形状，比如 `(1, -1)`，你应该使用 `reshape()` 方法，而不是 `view()`。

- `(1, -1)` 的含义是将数组变为 2D，其中第一维固定为 1，第二维自动计算（保持数据总量不变）。

**语法**：
```python
ndarray.reshape(new_shape)
```

**示例**：

```python
import numpy as np

# 创建一个一维数组
arr = np.array([1, 2, 3, 4])

# 调整形状为 (1, -1)
reshaped = arr.reshape(1, -1)

print("原始数组：", arr)
print("调整形状后的数组：", reshaped)
```

**输出**：
```
原始数组： [1 2 3 4]
调整形状后的数组： [[1 2 3 4]]
```

这里 `(1, -1)` 将数组调整为一个 2D 数组，第一维为 1，第二维由 NumPy 自动计算。

**总结**：

- **`view()`** 是用来创建一个共享数据内存的新视图，可以改变数据类型。
- **`reshape()`** 是用来调整数组形状的，如果你想用 `(1, -1)` 的形式，那应该使用 `reshape()` 而不是 `view()`。

如果你打算调整数组形状，请改用 `reshape`！

### ndarray[None:]

在NumPy中，`[None:]`用于数组的索引，表示在指定的轴上增加一个维度，并从该维度的起始位置到结束位置选择所有元素。

具体来说，[None:]可以分解为两部分：

1. `None`：在NumPy中，`None`通常用于表示一个新维度的开始。当在索引中使用`None`时，它会增加一个维度，使得数组的形状发生变化。
2. `:`：表示从当前维度的起始位置到结束位置选择所有元素。

### ndarray[..., [0, 2]]

在 NumPy 中，`ndarray[..., [0, 2]]` 的 `[0, 2]` 表示选择数组中所有元素的第 0 个和第 2 个索引对应的值。

具体来说，`ndarray[..., [0, 2]]` 可以分解为两部分：

1. `...`：表示省略号，用于选择数组中所有元素。在 NumPy 中，省略号可以用于多维数组的索引，表示选择所有维度上的所有元素。
2. `[0, 2]`：表示选择数组中每个元素的第 0 个和第 2 个索引对应的值。在 NumPy 中，这称为 fancy indexing，即使用一个整数数组来选择元素。

举个例子，假设我们有一个二维 NumPy 数组 `arr`：

```python
import numpy as np

arr = np.array(  # shape = (2, 3)
    [
        [1, 2, 3], 
        [4, 5, 6]
    ]
)
```

如果我们使用 `arr[..., [0, 2]]` 进行索引：

```python
arr[..., [0, 2]]
```

这将选择数组中每个元素的第 0 个和第 2 个索引对应的值。结果是一个二维数组，形状为 `(2, 2)`：

```
[[1 3]
 [4 6]]
```

在这个例子中，`[0, 2]` 表示选择每个子数组的第 0 个和第 2 个元素。

同样地，如果有一个三维数组 `arr3`：

```python
arr3 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])  # shape = (2, 2, 3)
```

使用 `arr3[..., [0, 2]]` 进行索引：

```python
arr3[..., [0, 2]]
```

这将选择数组中每个元素的第 0 个和第 2 个索引对应的值。结果是一个三维数组，形状为 `(2, 2, 2)`：

```
[[[ 1  3]
  [ 4  6]]

 [[ 7  9]
  [10 12]]]
```

在这个例子中，`[0, 2]` 表示选择每个子数组的第 0 个和第 2 个元素。

总结来说，`ndarray[..., [0, 2]]` 的 `[0, 2]` 表示选择数组中所有元素的第 0 个和第 2 个索引对应的值。这在处理多维数组时非常有用，可以方便地选择特定的元素。

# Tricks

## num_workers的计算方式

```python
import os
import torch

nd = torch.cuda.device_count()  # number of CUDA devices
nw = min(os.cpu_count() // max(nd, 1), workers)
```