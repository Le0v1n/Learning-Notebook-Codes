# 1. AMP 的定义

自动混合精度（Automatic Mixed Precision, AMP）训练是一种深度学习训练技术，它可以在训练过程中动态地选择使用浮点数的精度。

自动混合精度训练的基本思想是，根据计算的需求和成本，自动地在单精度（FP32）和半精度（FP16）之间切换。具体来说，AMP 训练会识别那些对精度要求不高的计算（例如，权重矩阵的乘法），并将这些计算转换为半精度（FP16）计算，以减少梯度计算中的数值误差。而对于那些对精度要求较高的计算（例如，激活函数的计算），AMP 训练仍然使用单精度（FP32）计算，以保持模型的准确性和响应性。

> - float32: 单精度浮点数
> - float16: 半精度浮点数
> - float64: 双精度浮点数

# 2. AMP 训练的优点

1. **提高训练速度**：使用双精度进行某些计算可以减少浮点运算的次数，从而提高训练速度。
2. **减少内存使用**：双精度通常需要比单精度更多的内存，但只在必要时使用双精度，可以减少总体内存使用。
3. **提高数值稳定性**：在一些情况下，使用双精度可以减少梯度更新的数值误差，提高模型的训练稳定性。

# 3. `torch.FloatTensor` 和 `torch.HalfTensor`

在 PyTorch 中，`torch.FloatTensor` 和 `torch.HalfTensor` 是两种不同精度的浮点张量类型，它们分别对应于单精度（FP32）和半精度（FP16）浮点数。

- **torch.FloatTensor**：这是 PyTorch 中的单精度浮点张量。它使用 32 位（4 字节）来存储每个浮点数，提供了较高的数值精度和较大的数值范围。这是大多数深度学习任务中默认使用的浮点类型。
- **torch.HalfTensor**：这是 PyTorch 中的半精度浮点张量。它使用 16 位（2 字节）来存储每个浮点数，数值范围和精度都比单精度浮点数低。然而，由于半精度浮点数占用的内存较少，因此在某些情况下（如内存受限的环境或需要大幅提高计算速度时）会使用半精度浮点数。

# 4. YOLOv5 中的 AMP

在 [Automatic mixed precision (AMP) training is now natively supported and a stable feature. #557](https://github.com/ultralytics/yolov5/issues/557) 有提到 AMP。

<div align=center>
    <img src=https://img-blog.csdnimg.cn/direct/a834990e08324b25a558c881a8737224.png
        width=100%>
    <center></center>
</div>

从图中可以看到，💡 YOLOv5 默认开启 AMP 训练，并且保存的模型也是 FP16 而非传统的 FP32。

# 5. 如何在 PyTorch 中使用 AMP？

```python
from torch.cuda.amp import Scaler, autocast
```

> ⚠️ 注意：
> 1. Scaler 并不是 AMP，autocast 也不是 AMP，只有 AMP + Scaler 才是 AMP
>
> 2. AMP 并不特指半精度，我们可以指定任意精度！

## 5.1 autocast

- 〔官方文档〕[torch.cuda.amp.autocast](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast)

### 5.1.1 定义

使用 `torch.cuda.amp` 模块中的 `autocast` 类。当进入 `autocast` 上下文后，支持 AMP 的 CUDA 算子会把 Tensor 的 `dtype` 转换为 FP16，从而在不损失训练精度的情况下加快运算。刚进入 `autocast` 的上下文时，Tensor 可以是任何类型，不需要在 `model` 或 `input` 上手工调用 `.half()`，框架会自动做，这就是 AMP 中的 Automatic。

另外需要注意的是，`autocast` 上下文应该只包含网络的前向推理过程（包括 loss 的计算），⚠️ 不要包含反向传播，因为 BP 的算子会使用和前向算子相同的类型。

### 5.1.2 代码

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


### 5.1.3 CUDA 设备的示例-1

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

### 5.1.4 CUDA 设备的示例-2

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

### 5.1.5 CPU 训练示例

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

## 5.2 GradScaler

- 〔官方文档〕[torch.cuda.amp.GradScaler](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler)

### 5.2.1 定义

使用 `torch.cuda.amp.GradScaler`，需要在训练最开始之前实例化一个 `GradScaler` 对象。通过放大 Loss 的值，从而防止梯度的 underflow（⚠️ 这只是 BP 的时候传递梯度信息使用，真正更新权重的时候还是要把放大的梯度再 unscale 回去）

### 5.2.2 代码

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


### 5.2.3 示例：典型的混合精度训练

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

### 5.2.4 示例：梯度累积

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

### 5.2.5 示例：处理多个模型、损失和优化器

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

### 5.2.6 示例：DataParallel (DP) in a single process

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

### 5.2.7 示例：DistributedDataParallel (DDP), 单卡单线程

`torch.nn.parallel.DistributedDataParallel` 的文档建议每个进程使用一个 GPU 以获得最佳性能。在这种情况下，`DistributedDataParallel` 不会在内部生成线程，因此对 autocast 和 GradScaler 的使用不受影响。

### 5.2.8 示例：DistributedDataParallel (DDP), 多卡多线程

在这里，`torch.nn.parallel.DistributedDataParallel` 可能会生成一个辅助线程来在每个设备上运行前向推理，类似于 `torch.nn.DataParallel`。

解决方法是相同的：在模型的前向方法中应用 autocast，以确保它在辅助线程中启用。