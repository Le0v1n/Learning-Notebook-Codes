# 0. 15 分钟上手 MMENGINE

以在 CIFAR-10 数据集上训练一个 ResNet-50 模型为例，我们将使用 80 行以内的代码，利用 MMEngine 构建一个完整的、可配置的训练和验证流程，整个流程包含如下步骤：

1. 构建模型
2. 构建数据集和数据加载器
3. 构建评测指标
4. 构建执行器并执行任务

## 0.1 构建模型

首先，我们需要构建一个模型，在 MMEngine 中，我们约定这个模型应当继承 `BaseModel`，并且其 `forward` 方法除了接受来自数据集的若干参数外，还需要接受额外的参数 `mode`：
- 对于训练，我们需要 `mode` 接受字符串 `"loss"`，并返回一个包含 `"loss"` 字段的字典；
- 对于验证，我们需要 `mode` 接受字符串 `"predict"`，并返回同时包含预测信息和真实信息的结果。

```python
import torch.nn.functional as F
import torchvision
from mmengine.model import BaseModel


class MMResNet50(BaseModel):  # 继承自BaseModel
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels
```

## 0.2 构建数据集和数据加载器

其次，我们需要构建训练和验证所需要的数据集 (Dataset) 和数据加载器 (DataLoader)。 对于基础的训练和验证功能，我们可以直接使用符合 PyTorch 标准的数据加载器和数据集。

```python
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
train_dataloader = DataLoader(batch_size=32,
                              shuffle=True,
                              dataset=torchvision.datasets.CIFAR10(
                                  'data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(**norm_cfg)
                                  ])))

val_dataloader = DataLoader(batch_size=32,
                            shuffle=False,
                            dataset=torchvision.datasets.CIFAR10(
                                'data/cifar10',
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(**norm_cfg)
                                ])))
```

## 0.3 构建评测指标

为了进行验证和测试，我们需要定义模型推理结果的评测指标。我们约定这一评测指标需要继承 `BaseMetric`，并实现 `process` 和 `compute_metrics` 方法。其中：

- `process` 方法接受数据集的输出和模型 `mode="predict"` 时的输出，此时的数据为一个批次的数据，对这一批次的数据进行处理后，保存信息至 `self.results` 属性。 
- `compute_metrics` 接受 `results` 参数，这一参数的输入为 `process` 中保存的所有信息 （如果是分布式环境，`results` 中为已收集的，包括各个进程 `process` 保存信息的结果），利用这些信息计算并返回保存有评测指标结果的字典。

```python
from mmengine.evaluator import BaseMetric


class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        # 将一个批次的中间结果保存至 `self.results`
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        # 返回保存有评测指标结果的字典，其中键为指标名称
        return dict(accuracy=100 * total_correct / total_size)
```

## 0.4 构建执行器并执行任务

最后，我们利用构建好的模型，数据加载器，评测指标构建一个执行器 (Runner)，同时在其中配置 优化器、工作路径、训练与验证配置等选项，即可通过调用 `train()` 接口启动训练：

```python
from torch.optim import SGD
from mmengine.runner import Runner


runner = Runner(
    # 用以训练和验证的模型，需要满足特定的接口需求
    model=MMResNet50(),

    # 工作路径，用以保存训练日志、权重文件信息
    work_dir='./work_dir',

    # 训练数据加载器，需要满足 PyTorch 数据加载器协议
    train_dataloader=train_dataloader,

    # 优化器包装，用于模型优化，并提供 AMP、梯度累积等附加功能
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),

    # 训练配置，用于指定训练周期、验证间隔等信息
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),

    # 验证数据加载器，需要满足 PyTorch 数据加载器协议
    val_dataloader=val_dataloader,

    # 验证配置，用于指定验证所需要的额外参数
    val_cfg=dict(),

    # 用于验证的评测器，这里使用默认评测器，并评测指标
    val_evaluator=dict(type=Accuracy),
)

runner.train()
```

最后，让我们把以上部分汇总成为一个完整的，利用 MMEngine 执行器进行训练和验证的脚本：

```python
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD
from torch.utils.data import DataLoader

from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner


class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels


class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        return dict(accuracy=100 * total_correct / total_size)


norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
train_dataloader = DataLoader(batch_size=32,
                              shuffle=True,
                              dataset=torchvision.datasets.CIFAR10(
                                  'data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(**norm_cfg)
                                  ])))

val_dataloader = DataLoader(batch_size=32,
                            shuffle=False,
                            dataset=torchvision.datasets.CIFAR10(
                                'data/cifar10',
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(**norm_cfg)
                                ])))

runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
)
runner.train()
```

输出的训练日志如下：

```
2022/08/22 15:51:53 - mmengine - INFO -
------------------------------------------------------------
System environment:
    sys.platform: linux
    Python: 3.8.12 (default, Oct 12 2021, 13:49:34) [GCC 7.5.0]
    CUDA available: True
    numpy_random_seed: 1513128759
    GPU 0: NVIDIA GeForce GTX 1660 SUPER
    CUDA_HOME: /usr/local/cuda
...

2022/08/22 15:51:54 - mmengine - INFO - Checkpoints will be saved to /home/mazerun/work_dir by HardDiskBackend.
2022/08/22 15:51:56 - mmengine - INFO - Epoch(train) [1][10/1563]  lr: 1.0000e-03  eta: 0:18:23  time: 0.1414  data_time: 0.0077  memory: 392  loss: 5.3465
2022/08/22 15:51:56 - mmengine - INFO - Epoch(train) [1][20/1563]  lr: 1.0000e-03  eta: 0:11:29  time: 0.0354  data_time: 0.0077  memory: 392  loss: 2.7734
2022/08/22 15:51:56 - mmengine - INFO - Epoch(train) [1][30/1563]  lr: 1.0000e-03  eta: 0:09:10  time: 0.0352  data_time: 0.0076  memory: 392  loss: 2.7789
2022/08/22 15:51:57 - mmengine - INFO - Epoch(train) [1][40/1563]  lr: 1.0000e-03  eta: 0:08:00  time: 0.0353  data_time: 0.0073  memory: 392  loss: 2.5725
2022/08/22 15:51:57 - mmengine - INFO - Epoch(train) [1][50/1563]  lr: 1.0000e-03  eta: 0:07:17  time: 0.0347  data_time: 0.0073  memory: 392  loss: 2.7382
2022/08/22 15:51:57 - mmengine - INFO - Epoch(train) [1][60/1563]  lr: 1.0000e-03  eta: 0:06:49  time: 0.0347  data_time: 0.0072  memory: 392  loss: 2.5956
2022/08/22 15:51:58 - mmengine - INFO - Epoch(train) [1][70/1563]  lr: 1.0000e-03  eta: 0:06:28  time: 0.0348  data_time: 0.0072  memory: 392  loss: 2.7351
...
2022/08/22 15:52:50 - mmengine - INFO - Saving checkpoint at 1 epochs
2022/08/22 15:52:51 - mmengine - INFO - Epoch(val) [1][10/313]    eta: 0:00:03  time: 0.0122  data_time: 0.0047  memory: 392
2022/08/22 15:52:51 - mmengine - INFO - Epoch(val) [1][20/313]    eta: 0:00:03  time: 0.0122  data_time: 0.0047  memory: 308
2022/08/22 15:52:51 - mmengine - INFO - Epoch(val) [1][30/313]    eta: 0:00:03  time: 0.0123  data_time: 0.0047  memory: 308
...
2022/08/22 15:52:54 - mmengine - INFO - Epoch(val) [1][313/313]  accuracy: 35.7000
```

基于 PyTorch 和基于 MMEngine 的训练流程对比如下：

<div align=center>
    <img src=./imgs_markdown/基于PyTorch和基于MMEngine的训练流程对比.gif
    width=100%>
    <center></center>
</div>

除了以上基础组件，你还可以利用执行器轻松地组合配置各种训练技巧，如开启混合精度训练和梯度累积（见 优化器封装（OptimWrapper））、配置学习率衰减曲线（见 评测指标与评测器（Metrics & Evaluator））等。

# 1. 执行器（RUNNER）

欢迎来到 MMEngine 用户界面的核心——执行器！

作为 MMEngine 中的“集大成者”，执行器涵盖了整个框架的方方面面，肩负着串联所有组件的重要责任；因此，其中的代码和实现逻辑需要兼顾各种情景，相对庞大复杂。但是不用担心！在这篇教程中，我们将隐去繁杂的细节，速览执行器常用的接口、功能、示例，为你呈现一个清晰易懂的用户界面。阅读完本篇教程，你将会：

1. 掌握执行器的常见参数与使用方式
2. 了解执行器的最佳实践——配置文件的写法
3. 了解执行器基本数据流与简要执行逻辑
4. 亲身感受使用执行器的优越性

## 1.2 执行器示例

使用执行器构建属于你自己的训练流程，通常有两种开始方式：

1. 参考 API 文档，逐项确认和配置参数
2. 在已有配置（如 15 分钟上手或 MMDet 等下游算法库）的基础上，进行定制化修改

两种方式各有利弊。使用前者，初学者很容易迷失在茫茫多的参数项中不知所措；而使用后者，一份过度精简或过度详细的参考配置都不利于初学者快速找到所需内容。

解决上述问题的关键在于，把执行器作为备忘录：掌握其中最常用的部分，并在有特殊需求时聚焦感兴趣的部分，其余部分使用缺省值。下面我们将通过一个适合初学者参考的例子，说明其中最常用的参数，并为一些不常用参数给出进阶指引。

### 1.2.1 面向初学者的示例代码

> 💡 我们希望你在本教程中更多地关注整体结构，而非具体模块的实现。这种“自顶向下”的思考方式是我们所倡导的。别担心，之后你将有充足的机会和指引，聚焦于自己想要改进的模块。

```python
# 导入 PyTorch 相关库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, default_collate
from torch.optim import Adam

# 导入MMEngine相关库
from mmengine.model import BaseModel
from mmengine.evaluator import BaseMetric
from mmengine.registry import MODELS, DATASETS, METRICS
from mmengine.runner import Runner


@MODELS.register_module()  # 注册模型
class MyAwesomeModel(BaseModel):
    def __init__(self, layers=4, activation='relu') -> None:
        super().__init__()
        if activation == 'relu':
            act_type = nn.ReLU
        elif activation == 'silu':
            act_type = nn.SiLU
        elif activation == 'none':
            act_type = nn.Identity
        else:
            raise NotImplementedError
        sequence = [nn.Linear(2, 64), act_type()]  # 至少有一层
        for _ in range(layers-1):  # 如果layers-1==0，那么就不执行了
            sequence.extend([nn.Linear(64, 64), act_type()])
        self.mlp = nn.Sequential(*sequence)  # 使用 nn.Sequential对list进行解包
        self.classifier = nn.Linear(64, 2)

    def forward(self, data, labels, mode):
        x = self.mlp(data)  # backbone
        x = self.classifier(x)  # classifier head
        if mode == 'tensor':  # 如果返回模型推理的结果
            return x
        elif mode == 'predict':  # 如果返回预测结果，则需要将其化为分数和对应的标签
            return F.softmax(x, dim=1), labels
        elif mode == 'loss':  # 如果返回损失，则直接使用x和标签做损失
            return {'loss': F.cross_entropy(x, labels)}


@DATASETS.register_module()  # 注册数据集
class MyDataset(Dataset):
    def __init__(self, is_train, size):
        self.is_train = is_train  # 判断此时是train还是val/test
        if self.is_train:
            torch.manual_seed(0)  # 设置随机数种子
            self.labels = torch.randint(0, 2, (size,))  # 随机生成标签
        else:
            torch.manual_seed(3407)  # Paper: 《seed(3407) is all you need》
            self.labels = torch.randint(0, 2, (size,))  # 随机生成标签
        
        # 随机生成数据
        r = 3 * (self.labels+1) + torch.randn(self.labels.shape)
        theta = torch.rand(self.labels.shape) * 2 * torch.pi
        self.data = torch.vstack([r*torch.cos(theta), r*torch.sin(theta)]).T

    def __getitem__(self, index):  # 传入索引就可以返回对应的data和label
        return self.data[index], self.labels[index]

    def __len__(self):  # 返回数据的长度
        return len(self.data)


@METRICS.register_module()  # 注册评估器
class Accuracy(BaseMetric):
    def __init__(self):
        super().__init__()

    def process(self, data_batch, data_samples):
        score, gt = data_samples  # 得到分数和对应的gt
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(r['correct'] for r in results)
        total_size = sum(r['batch_size'] for r in results)
        return dict(accuracy=100*total_correct/total_size)
    
    
if __name__ == "__main__":
    # 实例化Runner
    runner = Runner(
        # 你的模型
        model=MyAwesomeModel(
            layers=2,
            activation='relu'),
        
        # 模型检查点、日志等都将存储在工作路径中
        work_dir='exp/my_awesome_model',

        # 训练所用数据
        train_dataloader=DataLoader(
            dataset=MyDataset(
                is_train=True,
                size=10000),
            shuffle=True,
            collate_fn=default_collate,
            batch_size=64,
            pin_memory=True,
            num_workers=2),
        
        # 训练相关配置
        train_cfg=dict(
            by_epoch=True,   # 根据 epoch 计数而非 iteration
            max_epochs=10,
            val_begin=2,     # 从第 2 个 epoch 开始验证
            val_interval=1), # 每隔 1 个 epoch 进行一次验证

        # 优化器封装，MMEngine 中的新概念，提供更丰富的优化选择。
        # 通常使用默认即可，可缺省。有特殊需求可查阅文档更换，如
        # 'AmpOptimWrapper' 开启混合精度训练
        optim_wrapper=dict(
            optimizer=dict(
                type=Adam,
                lr=0.001)),
        # 参数调度器，用于在训练中调整学习率/动量等参数
        param_scheduler=dict(
            type='MultiStepLR',
            by_epoch=True,
            milestones=[4, 8],
            gamma=0.1),

        # 验证所用数据
        val_dataloader=DataLoader(
            dataset=MyDataset(
                is_train=False,
                size=1000),
            shuffle=False,
            collate_fn=default_collate,
            batch_size=1000,
            pin_memory=True,
            num_workers=2),
        # 验证相关配置，通常为空即可
        val_cfg=dict(),
        # 验证指标与验证器封装，可自由实现与配置
        val_evaluator=dict(type=Accuracy),

        # 以下为其他进阶配置，无特殊需要时尽量缺省
        # 钩子属于进阶用法，如无特殊需要，尽量缺省
        default_hooks=dict(
            # 最常用的默认钩子，可修改保存 checkpoint 的间隔
            checkpoint=dict(type='CheckpointHook', interval=1)),

        # `luancher` 与 `env_cfg` 共同构成分布式训练环境配置
        launcher='none',
        env_cfg=dict(
            cudnn_benchmark=False,   # 是否使用 cudnn_benchmark
            backend='nccl',   # 分布式通信后端
            mp_cfg=dict(mp_start_method='fork')),  # 多进程设置
        log_level='INFO',

        # 加载权重的路径 (None 表示不加载)
        load_from=None,
        # 从加载的权重文件中恢复训练
        resume=False
    )

    # 开始训练你的模型吧
    runner.train()
```

如果你通读了上述样例，即使不了解实现细节，你也一定大体理解了这个训练流程。MMEngine 期望的是：结构化、模块化、标准化的训练流程，使得复现更加可靠、对比更加清晰。

---

上述例子可能会让你产生如下问题：

<kbd><b>Question</b></kbd>：参数项实在是太多了！

<kbd><b>Answer</b></kbd>：不用担心，正如我们前面所说，把执行器Runner作为备忘录。执行器涵盖了方方面面，防止你漏掉重要内容，但是这并不意味着你需要配置所有参数。如15分钟上手中的极简例子（甚至，舍去 `val_evaluator` `val_dataloader` 和 `val_cfg`）也可以正常运行。所有的参数由你的需求驱动，**不关注的内容往往缺省值也可以工作得很好**。

---

<kbd><b>Question</b></kbd>：为什么有些传入参数是 dict？

<kbd><b>Answer</b></kbd>：是的，这与 MMEngine 的风格相关。在 MMEngine 中我们提供了两种不同风格的执行器构建方式：
1. 基于手动构建的
2. 基于注册机制的

下面的例子将给出一个对比：

```python
from mmengine.model import BaseModel
from mmengine.runner import Runner
from mmengine.registry import MODELS # 模型根注册器，你的自定义模型需要注册到这个根注册器中

@MODELS.register_module() # 用于注册的装饰器
class MyAwesomeModel(BaseModel): # 你的自定义模型
    def __init__(self, layers=18, activation='silu'):
        ...

# 基于注册机制的例子
runner = Runner(
    model=dict(
        type='MyAwesomeModel',
        layers=50,
        activation='relu'),
    ...
)

# 基于手动构建的例子
model = MyAwesomeModel(layers=18, activation='relu')
runner = Runner(
    model=model,
    ...
)
```

---

<kbd><b>Question</b></kbd>：我应该去哪里找到 xxx 参数的可能配置选项？

<kbd><b>Answer</b></kbd>：你可以在对应模块的教程中找到丰富的说明和示例，你也可以在 [API 文档](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.runner.Runner.html#mmengine.runner.Runner) 中找到 Runner 的所有参数。

---

<kbd><b>Question</b></kbd>：我来自 MMDet/MMCls...下游库，为什么例子写法与我接触的不同？

<kbd><b>Answer</b></kbd>：OpenMMLab 下游库广泛采用了配置文件的方式。我们将在下个章节，基于上述示例稍微变换，从而展示配置文件 MMEngine 中执行器的最佳实践的用法。

## 1.3 执行器最佳实践——配置文件

MMEngine 提供了一套支持 Python 语法的、功能强大的配置文件系统。你可以从之前的示例代码中近乎（我们将在下面说明）无缝地转换到配置文件。下面给出一段示例代码：

```python
# 以下代码存放在 example_config.py 文件中
# 基本拷贝自上面的示例，并将每项结尾的逗号删去
model = dict(
    type='MyAwesomeModel',
    layers=2,
    activation='relu')
work_dir = 'exp/my_awesome_model'

train_dataloader = dict(
    dataset=dict(
        type='MyDataset',
        is_train=True,
        size=10000),
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),
    collate_fn=dict(type='default_collate'),
    batch_size=64,
    pin_memory=True,
    num_workers=2)

train_cfg = dict(
    by_epoch=True,
    max_epochs=10,
    val_begin=2,
    val_interval=1)

optim_wrapper = dict(
    optimizer=dict(
        type='Adam',
        lr=0.001))
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[4, 8],
    gamma=0.1)

val_dataloader = dict(
    dataset=dict(type='MyDataset',
        is_train=False,
        size=1000),
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),
    collate_fn=dict(type='default_collate'),
    batch_size=1000,
    pin_memory=True,
    num_workers=2)
val_cfg = dict()
val_evaluator = dict(type='Accuracy')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1))
launcher = 'none'
env_cfg = dict(
    cudnn_benchmark=False,
    backend='nccl',
    mp_cfg=dict(mp_start_method='fork'))
log_level = 'INFO'
load_from = None
resume = False
```

此时，我们只需要在训练代码中加载配置，然后运行即可：

```python
from mmengine.config import Config
from mmengine.runner import Runner


# 加载指定位置的配置文件
config = Config.fromfile('example_config.py')
runner = Runner.from_cfg(config)
runner.train()
```

> ⚠️ **注意**：
> 
> - 虽然是 Python 语法，但合法的配置文件需要满足以下条件：所有的变量必须是基本类型（例如 `str` `dict` `int` 等）。因此，<b>配置文件系统高度依赖于注册机制，以实现从基本类型到其他类型（如 `nn.Module`）的构建</b>。
>
> - 使用配置文件时，你通常不需要手动注册所有模块。例如，`torch.optim` 中的所有优化器（如 `Adam` `SGD` 等）都已经在 `mmengine.optim` 中注册完成。使用时的经验法则是：**尝试直接使用 PyTorch 中的组件，只有当出现报错时再手动注册**。
>
> - 当使用配置文件写法时，你的自定义模块的实现代码通常存放在独立文件中，可能并未被正确注册，进而导致构建失败。我们推荐你阅读配置文档中 `custom_imports` 相关的内容以更好地使用配置文件系统。

执行器配置文件已经在 OpenMMLab 的众多下游库（MMCls，MMDet…）中被广泛使用，并成为事实标准与最佳实践。配置文件的功能远不止如此，如果你对于继承、覆写等进阶功能感兴趣，请参考[配置（Config）文档](https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/config.html)。

## 1.4 基本数据流<a id='基本数据流'></a>

在本章节中，我们将会介绍执行器内部各模块之间的数据传递流向与格式约定。如果你还没有基于 MMEngine 构建一个训练流程，本章节的部分内容可能会比较抽象、枯燥；你也可以暂时跳过，并在将来有需要时结合实践进行阅读。

接下来，我们将稍微深入执行器的内部，结合图示来理清其中数据的流向与格式约定。

<div align=center>
    <img src=./imgs_markdown/2024-02-27-09-34-28.png
    width=100%>
    <center></center>
</div>

上图是执行器的基本数据流，其中虚线边框、灰色填充的不同形状代表不同的数据格式，实线方框代表模块或方法。由于 MMEngine 强大的灵活性与可扩展性，你总可以继承某些关键基类并重载其中的方法，因此上图并不总是成立。只有当你没有自定义 `Runner` 或 `TrainLoop` ，并且你的自定义模型没有重载 `train_step`、`val_step` 与 `test_step` 方法时上图才会成立（而这在检测、分割等任务上是常见的，参考[模型](https://mmengine.readthedocs.io/zh-cn/latest/tutorials/model.html)教程）。

---

<kbd><b>Question</b></kbd>：可以确切地说明图中传递的每项数据的具体类型吗？

<kbd><b>Answer</b></kbd>：很遗憾，这一点无法做到。虽然 MMEngine 做了大量类型注释，但 Python 是一门高度动态化的编程语言，同时以数据为核心的深度学习系统也需要足够的灵活性来处理纷繁复杂的数据源，你有充分的自由决定何时需要（有时是必须）打破类型约定。因此，在你自定义某一或某几个模块（如 val_evaluator ）时，你需要确保它的输入与上游（如 model 的输出）兼容，同时输出可以被下游解析。MMEngine 将处理数据的灵活性交给了用户，因而也需要用户保证数据流的兼容性——当然，实际上手后会发现，这一点并不十分困难。

数据一致性的考验一直存在于深度学习领域，MMEngine 也在尝试用自己的方式改进。如果你有兴趣，可以参考数据集基类与抽象数据接口文档——但是请注意，它们主要面向进阶用户。

---

<kbd><b>Question</b></kbd>：`dataloader`、`model` 和 `evaluator` 之间的数据格式是如何约定的？

<kbd><b>Answer</b></kbd>：针对图中所展示的基本数据流，上述三个模块之间的数据传递可以用如下伪代码表示：

```python
# 训练过程
for data_batch in train_dataloader:
    data_batch = data_preprocessor(data_batch)
    if isinstance(data_batch, dict):
        losses = model.forward(**data_batch, mode='loss')
    elif isinstance(data_batch, (list, tuple)):
        losses = model.forward(*data_batch, mode='loss')
    else:
        raise TypeError()

# 验证过程
for data_batch in val_dataloader:
    data_batch = data_preprocessor(data_batch)
    if isinstance(data_batch, dict):
        outputs = model.forward(**data_batch, mode='predict')
    elif isinstance(data_batch, (list, tuple)):
        outputs = model.forward(**data_batch, mode='predict')
    else:
        raise TypeError()
    evaluator.process(data_samples=outputs, data_batch=data_batch)
metrics = evaluator.evaluate(len(val_dataloader.dataset))
```

上述伪代码的关键点在于：

- `data_preprocessor` 的输出需要经过解包后传递给 `model`
- `evaluator` 的 `data_samples` 参数接收模型的预测结果，而 `data_batch` 参数接收 `dataloader` 的原始数据

---

<kbd><b>Question</b></kbd>：什么是 `data_preprocessor`？我可以用它做裁减缩放等图像预处理吗？

<kbd><b>Answer</b></kbd>：虽然图中的 `data preprocessor` 与 `model` 是分离的，但在实际中前者是后者的一部分，因此可以在模型文档中的数据处理器章节找到。

<b>通常来说，数据处理器不需要额外关注和指定，默认的数据处理器只会自动将数据搬运到 GPU 中</b>。但是，如果你的模型与数据加载器的数据格式不匹配，你也可以自定义一个数据处理器来进行格式转换。

裁减缩放等图像预处理更推荐在[数据变换](https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/data_transform.html)中进行，但如果是 `batch` 相关的数据处理（如 `batch-resize` 等），可以在这里实现。

---

<kbd><b>Question</b></kbd>：为什么 `model` 产生了 3 个不同的输出？ `loss`、`predict`、`tensor` 是什么含义？

<kbd><b>Answer</b></kbd>：[15 分钟上手](https://mmengine.readthedocs.io/zh-cn/latest/get_started/15_minutes.html)对此有一定的描述，<b>你需要在自定义模型的 `forward` 函数中实现 3 条数据通路，适配训练、验证等不同需求</b>。模型文档中对此有详细解释。

---

<kbd><b>Question</b></kbd>：我可以看出红线是训练流程，蓝线是验证/测试流程，但绿线是什么？

<kbd><b>Answer</b></kbd>：在目前的执行器流程中，`'tensor'` 模式的输出并未被使用，大多数情况下用户无需实现。但一些情况下输出中间结果可以方便地进行 Debug。

---

<kbd><b>Question</b></kbd>：如果我重载了 train_step 等方法，上图会完全失效吗？

<kbd><b>Answer</b></kbd>：默认的 `train_step`、`val_step`、`test_step` 的行为，覆盖了从数据进入 `data preprocessor` 到 `model` 输出 `loss`、`predict` 结果的这一段流程，不影响其余部分。

## 1.5 为什么使用执行器（可选）

> 💡 这一部分内容并不能教会你如何使用执行器乃至整个 MMEngine，如果你正在被雇主/教授/DDL催促着几个小时内拿出成果，那这部分可能无法帮助到你，请随意跳过。但我们仍强烈推荐抽出时间阅读本章节，这可以帮助你更好地理解并使用 MMEngine

执行器是 MMEngine 中所有模块的“管理者”。所有的独立模块——不论是模型、数据集这些看得见摸的着的，还是日志记录、分布式训练、随机种子等相对隐晦的——都在执行器中被统一调度、产生关联。事物之间的关系是复杂的，但执行器为你处理了一切，并提供了一个清晰易懂的配置式接口。这样做的好处主要有：

1. 你可以轻易地在已搭建流程上修改/添加所需配置，而不会搅乱整个代码。也许你起初只有单卡训练，但你随时可以添加1、2行的分布式配置，切换到多卡甚至多机训练
2. 你可以享受 MMEngine 不断引入的新特性，而不必担心后向兼容性。混合精度训练、可视化、崭新的分布式训练方式、多种设备后端……我们会在保证后向兼容性的前提下不断吸收社区的优秀建议与前沿技术，并以简洁明了的方式提供给你
3. 你可以集中关注并实现自己的惊人想法，而不必受限于其他恼人的、不相关的细节。执行器的缺省值会为你处理绝大多数的情况

所以，MMEngine 与执行器会确实地让你更加轻松。只要花费一点点努力完成迁移，你的代码与实验会随着 MMEngine 的发展而与时俱进；如果再花费一点努力，MMEngine 的配置系统可以让你更加高效地管理数据、模型、实验。便利性与可靠性，这些正是我们努力的目标。

# 2. 数据集（DATASET）与数据加载器（DATALOADER）

> 💡 如果你没有接触过 PyTorch 的数据集与数据加载器，我们推荐先浏览 [PyTorch 官方教程](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)以了解一些基本概念

## 2.1 Dataset和Dataloader的介绍

数据集与数据加载器是 MMEngine 中训练流程的必要组件，它们的概念来源于 PyTorch，并且在含义上与 PyTorch 保持一致。通常来说，数据集定义了数据的总体数量、读取方式以及预处理，而数据加载器则在不同的设置下迭代地加载数据，如批次大小（`batch_size`）、随机乱序（`shuffle`）、并行（`num_workers`）等。数据集经过数据加载器封装后构成了数据源。在本篇教程中，我们将按照从外（数据加载器）到内（数据集）的顺序，逐步介绍它们在 MMEngine 执行器中的用法，并给出一些常用示例。读完本篇教程，你将会：

1. 掌握如何在 MMEngine 的执行器中配置数据加载器
2. 学会在配置文件中使用已有（如 torchvision）数据集
3. 了解如何使用自己的数据集

## 2.2 数据加载器详解

在执行器（Runner）中，你可以分别配置以下 3 个参数来指定对应的数据加载器

1. `train_dataloader`：在 `Runner.train()` 中被使用，为模型提供训练数据
2. `val_dataloader`：在 `Runner.val()` 中被使用，也会在 `Runner.train()` 中每间隔一段时间被使用，用于模型的验证评测
3. `test_dataloader`：在 `Runner.test()` 中被使用，用于模型的测试

MMEngine 完全支持 PyTorch 的原生 `DataLoader`，因此上述 3 个参数均可以直接传入构建好的 DataLoader，如 15 分钟上手中的例子所示。同时，借助 MMEngine 的注册机制，以上参数也可以传入 `dict`，如下面代码（以下简称例 1）所示。字典中的键值与 `DataLoader` 的构造参数一一对应。

```python
runner = Runner(
    train_dataloader=dict(
        batch_size=32,
        sampler=dict(
            type='DefaultSampler',
            shuffle=True),
        dataset=torchvision.datasets.CIFAR10(...),
        collate_fn=dict(type='default_collate')
    )
)
```

在这种情况下，数据加载器会在实际被用到时，在执行器内部被构建。

> 💡 Tips：
>
> - 关于 DataLoader 的更多可配置参数，你可以参考 [PyTorch API 文档](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
> - 如果你对于构建的具体细节感兴趣，你可以参考 [build_dataloader](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.runner.Runner.html#mmengine.runner.Runner.build_dataloader)

细心的你可能会发现，例 1 并非直接由 15 分钟上手中的示例代码简单修改而来。你可能本以为将 `DataLoader` 简单替换为 `dict` 就可以无缝切换，但遗憾的是，基于注册机制构建时 MMEngine 会有一些隐式的转换和约定。我们将介绍其中的不同点，以避免你使用配置文件时产生不必要的疑惑。

### 2.2.1 sampler 与 shuffle

与 15 分钟上手明显不同，例 1 中我们添加了 `sampler` 参数，这是由于在 MMEngine 中我们要求通过 dict 传入的数据加载器的配置必须包含 `sampler` 参数。同时，`shuffle` 参数也从 `DataLoader` 中移除，这是由于在 PyTorch 中 `sampler` 与 `shuffle` 参数是互斥的，见 PyTorch API 文档。

> 事实上，在 PyTorch 的实现中，`shuffle` 只是一个便利记号。当设置为 `True` 时 `DataLoader` 会自动在内部使用 `RandomSampler`

当考虑 `sampler` 时，例 1 代码基本可以认为等价于下面的代码块：

```python
from mmengine.dataset import DefaultSampler


dataset = torchvision.datasets.CIFAR10(...)
sampler = DefaultSampler(dataset, shuffle=True)

runner = Runner(
    train_dataloader=DataLoader(  # 传入的不是一个dict，而是一个Dataloader对象
        batch_size=32,
        sampler=sampler,
        dataset=dataset,
        collate_fn=default_collate
    )
)
```

> ⚠️  上述代码的等价性只有在：① 使用单进程训练，以及 ② 没有配置执行器的 `randomness` 参数时成立。这是由于使用 `dict` 传入 `sampler` 时，执行器会保证它在分布式训练环境设置完成后才被惰性构造，并接收到正确的随机种子。这两点在手动构造时需要额外工作且极易出错。因此，上述的写法只是一个示意而非推荐写法。我们强烈建议 `sampler` 以 `dict` 的形式传入，让执行器处理构造顺序，以避免出现问题。

### 2.2.2 DefaultSampler

上面例子可能会让你好奇：`DefaultSampler` 是什么，为什么要使用它，是否有其他选项？事实上，`DefaultSampler` 是 MMEngine 内置的一种采样器，它屏蔽了单进程训练与多进程训练的细节差异，使得单卡与多卡训练可以无缝切换。如果你有过使用 PyTorch `DistributedDataParallel` (DDP) 的经验，你一定会对其中更换数据加载器的 `sampler` 参数有所印象。但在 MMEngine 中，这一细节通过 `DefaultSampler` 而被屏蔽。

除了 `Dataset` 本身之外，`DefaultSampler` 还支持以下参数配置：

- `shuffle` 设置为 `True` 时会打乱数据集的读取顺序
- `seed` 打乱数据集所用的随机种子，通常不需要在此手动设置，会从 `Runner` 的 `randomness` 入参中读取
- `round_up` 设置为 `True` 时，与 PyTorch DataLoader 中设置 `drop_last=False` 行为一致。如果你在迁移 PyTorch 的项目，你可能需要注意这一点。

> 更多关于 DefaultSampler 的内容可以参考 [API 文档](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.dataset.DefaultSampler.html#mmengine.dataset.DefaultSampler)

`DefaultSampler` 适用于绝大部分情况，并且我们保证在执行器中使用它时，随机数等容易出错的细节都被正确地处理，防止你陷入多进程训练的常见陷阱。如果你想要使用基于迭代次数 (iteration-based) 的训练流程，你也许会对 [`InfiniteSampler`](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.dataset.InfiniteSampler.html#mmengine.dataset.InfiniteSampler) 感兴趣。如果你有更多的进阶需求，你可能会想要参考上述两个内置 `sampler` 的代码，实现一个自定义的 `sampler` 并注册到 `DATA_SAMPLERS` 根注册器中。

```python
@DATA_SAMPLERS.register_module()
class MySampler(Sampler):
    pass

runner = Runner(
    train_dataloader=dict(
        sampler=dict(type='MySampler'),
        ...
    )
)
```

### 2.2.3 不起眼的 collate_fn

PyTorch 的 `DataLoader` 中，`collate_fn` 这一参数常常被使用者忽略，但在 MMEngine 中你需要额外注意：当你传入 `dict` 来构造数据加载器时，MMEngine 会默认使用内置的 `pseudo_collate`，这一点明显区别于 PyTorch 默认的 `default_collate`。因此，当你迁移 PyTorch 项目时，需要在配置文件中手动指明 `collate_fn` 以保持行为一致。

> MMEngine 中使用 `pseudo_collate` 作为默认值，主要是由于历史兼容性原因，你可以不必过于深究，只需了解并避免错误使用即可。

MMengine 中提供了 2 种内置的 `collate_fn`：

- `pseudo_collate`，缺省时的默认参数。它不会将数据沿着 `batch` 的维度合并。详细说明可以参考 [pseudo_collate](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.dataset.pseudo_collate.html#mmengine.dataset.pseudo_collate)
- `default_collate`，与 PyTorch 中的 `default_collate` 行为几乎完全一致，会将数据转化为 Tensor 并沿着 batch 维度合并。一些细微不同和详细说明可以参考 [default_collate](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.dataset.default_collate.html#mmengine.dataset.default_collate)

如果你想要使用自定义的 `collate_fn`，你也可以将它注册到 `FUNCTIONS` 根注册器中来使用：

```python
@FUNCTIONS.register_module()
def my_collate_func(data_batch: Sequence) -> Any:
    pass

runner = Runner(
    train_dataloader=dict(
        ...
        collate_fn=dict(type='my_collate_func')
    )
)
```

## 2.3 数据集详解

数据集通常定义了数据的数量、读取方式与预处理，并作为参数传递给数据加载器供后者分批次加载。由于我们使用了 PyTorch 的 `DataLoader`，因此数据集也自然与 PyTorch `Dataset` 完全兼容。同时得益于注册机制，当数据加载器使用 `dict` 在执行器内部构建时，`dataset` 参数也可以使用 `dict` 传入并在内部被构建。这一点使得编写配置文件成为可能。

### 2.3.1 使用 torchvision 数据集

`torchvision` 中提供了丰富的公开数据集，它们都可以在 MMEngine 中直接使用，例如 15 分钟上手中的示例代码就使用了其中的 `Cifar10` 数据集，并且使用了 `torchvision` 中内置的数据预处理模块。

但是，当需要将上述示例转换为配置文件时，你需要对 `torchvision` 中的数据集进行额外的注册。如果你同时用到了 `torchvision` 中的数据预处理模块，那么你也需要编写额外代码来对它们进行注册和构建。下面我们将给出一个等效的例子来展示如何做到这一点。

```python
import torchvision.transforms as tvt
from mmengine.registry import DATASETS, TRANSFORMS
from mmengine.dataset.base_dataset import Compose


# 注册 torchvision 的 CIFAR10 数据集
# 数据预处理也需要在此一起构建
@DATASETS.register_module(name='Cifar10', force=False)
def build_torchvision_cifar10(transform=None, **kwargs):
    if isinstance(transform, dict):
        transform = [transform]
    if isinstance(transform, (list, tuple)):
        transform = Compose(transform)
    return torchvision.datasets.CIFAR10(**kwargs, transform=transform)

# 注册 torchvision 中用到的数据预处理模块
DATA_TRANSFORMS.register_module('RandomCrop', module=tvt.RandomCrop)
DATA_TRANSFORMS.register_module('RandomHorizontalFlip', module=tvt.RandomHorizontalFlip)
DATA_TRANSFORMS.register_module('ToTensor', module=tvt.ToTensor)
DATA_TRANSFORMS.register_module('Normalize', module=tvt.Normalize)

# 在 Runner 中使用
runner = Runner(
    train_dataloader=dict(
        batch_size=32,
        sampler=dict(
            type='DefaultSampler',
            shuffle=True),
        dataset=dict(type='Cifar10',
            root='data/cifar10',
            train=True,
            download=True,
            transform=[
                dict(type='RandomCrop', size=32, padding=4),
                dict(type='RandomHorizontalFlip'),
                dict(type='ToTensor'),
                dict(type='Normalize', **norm_cfg)])
    )
)
```

> 💡  上述例子中大量使用了注册机制，并且用到了 MMEngine 中的 `Compose`。如果你急需在配置文件中使用 `torchvision` 数据集，你可以参考上述代码并略作修改。但我们更加推荐你有需要时在下游库（如 MMDet 和 MMPretrain 等）中寻找对应的数据集实现，从而获得更好的使用体验。

### 2.3.2 自定义数据集

你可以像使用 PyTorch 一样，自由地定义自己的数据集，或将之前 PyTorch 项目中的数据集拷贝过来。如果你想要了解如何自定义数据集，可以参考 PyTorch 官方教程。

### 2.3.3 使用 MMEngine 的数据集基类

除了直接使用 PyTorch 的 `Dataset` 来自定义数据集之外，你也可以使用 MMEngine 内置的 `BaseDataset`，参考数据集基类文档。它对标注文件的格式做了一些约定，使得数据接口更加统一、多任务训练更加便捷。同时，数据集基类也可以轻松地搭配内置的数据变换使用，减轻你从头搭建训练流程的工作量。

目前，`BaseDataset` 已经在 OpenMMLab 2.0 系列的下游仓库中被广泛使用。

# 3. 模型（MODEL）

## 3.1 Runner 与 model

在 [Runner 教程的基本数据流](#基本数据流)中我们提到，DataLoader、model 和 evaluator 之间的数据流通遵循了一些规则，我们先来回顾一下基本数据流的伪代码：

```python
# 训练过程
for data_batch in train_dataloader:
    data_batch = model.data_preprocessor(data_batch, training=True)
    if isinstance(data_batch, dict):
        losses = model(**data_batch, mode='loss')
    elif isinstance(data_batch, (list, tuple)):
        losses = model(*data_batch, mode='loss')
    else:
        raise TypeError()

# 验证过程
for data_batch in val_dataloader:
    data_batch = model.data_preprocessor(data_batch, training=False)
    if isinstance(data_batch, dict):
        outputs = model(**data_batch, mode='predict')
    elif isinstance(data_batch, (list, tuple)):
        outputs = model(**data_batch, mode='predict')
    else:
        raise TypeError()
    evaluator.process(data_samples=outputs, data_batch=data_batch)

metrics = evaluator.evaluate(len(val_dataloader.dataset))
```

在 `Runner` 的教程中，我们简单介绍了模型和前后组件之间的数据流通关系，提到了 `data_preprocessor` 的概念，对 `model` 有了一定的了解。然而在 `Runner` 实际运行的过程中，模型的功能和调用关系，其复杂程度远超上述伪代码。为了让你能够不感知模型和外部组件的复杂关系，进而聚焦精力到算法本身，我们设计了 `BaseModel`。大多数情况下你只需要让 `model` 继承 `BaseModel`，并按照要求实现 `forward` 接口，就能完成训练、测试、验证的逻辑。

在继续阅读模型教程之前，我们先抛出两个问题，希望你在阅读完 model 教程后能够找到相应的答案：

1. 我们在什么位置更新模型参数？如果我有一些非常复杂的参数更新逻辑，又该如何实现？
2. 为什么要有 `data_preprocessor` 的概念？它又可以实现哪些功能？

## 3.2 接口约定

在训练深度学习任务时，我们通常需要定义一个模型来实现算法的主体。在基于 MMEngine 开发时，定义的模型由执行器管理，且需要实现 `train_step`、`val_step` 和 `test_step` 方法。 对于检测、识别、分割一类的深度学习任务，上述方法通常为标准的流程，例如：

- 在 `train_step` 里更新参数，返回损失；
- 在 `val_step` 和 `test_step` 里返回预测结果。

因此 MMEngine 抽象出模型基类 `BaseModel`，实现了上述接口的标准流程。

得益于 `BaseModel` 我们只需要让模型继承自模型基类，并按照一定的规范实现 `forward`，就能让模型在执行器中运行起来。

> 💡  模型基类继承自[模块基类](https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/initialize.html)，能够通过配置 `init_cfg` 灵活地选择初始化方式。

〔[forward](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.model.BaseModel.html#mmengine.model.BaseModel.forward)〕：`forward` 的入参需通常需要和 DataLoader 的输出保持一致 (自定义数据预处理器除外)，如果 `DataLoader` 返回元组类型的数据 `data`，`forward` 需要能够接受 `*data` 的解包后的参数；如果返回字典类型的数据 `data`，`forward` 需要能够接受 `**data` 解包后的参数。 `mode` 参数用于控制 `forward` 的返回结果：

- `mode='loss'`：loss 模式通常在训练阶段启用，并返回一个损失字典。损失字典的 key-value 分别为损失名和可微的 `torch.Tensor`。字典中记录的损失会被用于更新参数和记录日志。模型基类会在 `train_step` 方法中调用该模式的 `forward`。

- `mode='predict'`： predict 模式通常在验证、测试阶段启用，并返回列表/元组形式的预测结果，预测结果需要和 `process` 接口的参数相匹配。OpenMMLab 系列算法对 predict 模式的输出有着更加严格的约定，需要输出列表形式的数据元素。模型基类会在 `val_step``，test_step` 方法中调用该模式的 `forward`。

- `mode='tensor'`：tensor 和 predict 模式均返回模型的前向推理结果，区别在于 tensor 模式下，forward 会返回未经后处理的张量，例如返回未经非极大值抑制（nms）处理的检测结果，返回未经 argmax 处理的分类结果。我们可以基于 tensor 模式的结果进行自定义的后处理。

〔[train_step](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.model.BaseModel.html#mmengine.model.BaseModel.train_step)〕执行 forward 方法的 loss 分支，得到损失字典。模型基类基于优化器封装 实现了标准的梯度计算、参数更新、梯度清零流程。其等效伪代码如下：

```python
def train_step(self, data, optim_wrapper):
    data = self.data_preprocessor(data, training=True)  # 按下不表，详见数据与处理器一节
    loss = self(**data, mode='loss')  # loss 模式，返回损失字典，假设 data 是字典，使用 ** 进行解析。事实上 train_step 兼容 tuple 和 dict 类型的输入。
    parsed_losses, log_vars = self.parse_losses() # 解析损失字典，返回可以 backward 的损失以及可以被日志记录的损失
    optim_wrapper.update_params(parsed_losses)  # 更新参数
    return log_vars
```

〔[val_step](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.model.BaseModel.html#mmengine.model.BaseModel.val_step)〕执行 forward 方法的 predict 分支，返回预测结果：

```python
def val_step(self, data, optim_wrapper):
    data = self.data_preprocessor(data, training=False)
    outputs = self(**data, mode='predict') # 预测模式，返回预测结果
    return outputs
```

〔[test_step](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.model.BaseModel.html#mmengine.model.BaseModel.test_step)〕同 val_step

看到这我们就可以给出一份 基本数据流伪代码 plus：

```python
# 训练过程
for data_batch in train_dataloader:
    loss_dict = model.train_step(data_batch)
# 验证过程
for data_batch in val_dataloader:
    preds = model.test_step(data_batch)
    evaluator.process(data_samples=outputs, data_batch=data_batch)
metrics = evaluator.evaluate(len(val_dataloader.dataset))
```

没错，抛开 Hook 不谈，loop 调用 model 的过程和上述代码一模一样！看到这，我们再回过头去看 15 分钟上手 MMEngine 里的模型定义部分，就有一种看山不是山的感觉：

```python
import torch.nn.functional as F
import torchvision
from mmengine.model import BaseModel


class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels

    # 下面的 3 个方法已在 BaseModel 实现，这里列出是为了
    # 解释调用过程
    def train_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data)
        loss = self(*data, mode='loss')  # CIFAR10 返回 tuple，因此用 * 解包
        parsed_losses, log_vars = self.parse_losses()
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data)
        outputs = self(*data, mode='predict')
        return outputs

    def test_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data)
        outputs = self(*data, mode='predict')
        return outputs
```

看到这里，相信你对数据流有了更加深刻的理解，也能够回答 Runner 与 model 里提到的第一个问题：

`BaseModel.train_step` 里实现了默认的参数更新逻辑，如果我们想实现自定义的参数更新流程，可以重写 `train_step` 方法。但是需要注意的是，我们需要保证 `train_step` 最后能够返回损失字典。

## 3.3 数据预处理器（DataPreprocessor）

如果你的电脑配有 GPU（或其他能够加速训练的硬件，如 MPS、IPU 等），并且运行了 15 分钟上手 MMEngine 的代码示例，你会发现程序是在 GPU 上运行的，那么 MMEngine 是在何时把数据和模型从 CPU 搬运到 GPU 的呢？

事实上，执行器会在构造阶段将模型搬运到指定设备，而数据则会在上一节提到的 `self.data_preprocessor` 这一行搬运到指定设备，进一步将处理好的数据传给模型。看到这里相信你会疑惑：

1. `MMResNet50` 并没有配置 `data_preprocessor`，为什么却可以访问到 `data_preprocessor`，并且把数据搬运到 GPU？
2. 为什么不直接在模型里调用 `data.to(device)` 搬运数据，而需要有 `data_preprocessor` 这一层抽象？它又能实现哪些功能？

首先回答第一个问题：`MMResNet50` 继承了 `BaseModel`。在执行 `super().__init__` 时，如果不传入任何参数，会构造一个默认的 `BaseDataPreprocessor`，其等效简易实现如下：

```python
class BaseDataPreprocessor(nn.Module):
    def forward(self, data, training=True):  # 先忽略 training 参数
        # 假设 data 是 CIFAR10 返回的 tuple 类型数据，事实上
        # BaseDataPreprocessor 可以处理任意类型的数
        # BaseDataPreprocessor 同样可以把数据搬运到多种设备，这边方便
        # 起见写成 .cuda()
        return tuple(_data.cuda() for _data in data)
```

`BaseDataPreprocessor` 会在训练过程中，将各种类型的数据搬运到指定设备。

在回答第二个问题之前，我们不妨先再思考几个问题

<kbd><b>Question</b></kbd>：数据归一化操作应该在哪里进行，`transform` 还是 `model`？

<kbd><b>Answer</b></kbd>：听上去好像都挺合理，放在 `transform` 里可以利用 `Dataloader` 的多进程加速，放在 `model` 里可以搬运到 GPU 上，利用 GPU 资源加速归一化。然而在我们纠结 CPU 归一化快还是 GPU 归一化快的时候，CPU 到 GPU 的数据搬运耗时相较于前者，可算的上是“降维打击”。 事实上对于归一化这类计算量较低的操作，其耗时会远低于数据搬运，因此优化数据搬运的效率就显得更加重要。设想一下，如果我能够在数据仍处于 `uint8` 时、归一化之前将其搬运到指定设备上（归一化后的 `float` 型数据大小是 `unit8` 的 4 倍），就能降低带宽，大大提升数据搬运的效率。这种“滞后”归一化的行为，也是我们设计数据预处理器（data preprocessor） 的主要原因之一。数据预处理器会先搬运数据，再做归一化，提升数据搬运的效率。

---

<kbd><b>Question</b></kbd>：我们应该如何实现 `MixUp`、`Mosaic` 一类的数据增强？

<kbd><b>Answer</b></kbd>：尽管看上去 `MixUp` 和 `Mosaic` 只是一种特殊的数据变换，按理说应该在 `transform` 里实现。考虑到这两种增强会涉及到“将多张图片融合成一张图片”的操作，在 `transform` 里实现他们的难度就会很大，因为目前 `transform` 的范式是对一张图片做各种增强，我们很难在一个 `transform` 里去额外读取其他图片（`transform` 里无法访问到 `dataset`）。然而如果基于 `Dataloader` 采样得到的 `batch_data` 去实现 `Mosaic` 或者 `Mixup`，事情就会变得非常简单，因为这个时候我们能够同时访问多张图片，可以轻而易举的完成图片融合的操作：

```python
class MixUpDataPreprocessor(nn.Module):
    def __init__(self, num_class, alpha):
        self.alpha = alpha

    def forward(self, data, training=True):
        data = tuple(_data.cuda() for _data in data)
        # 验证阶段无需进行 MixUp 数据增强
        if not training:
            return data

        label = F.one_hot(label)  # label 转 onehot 编码
        batch_size = len(label)
        index = torch.randperm(batch_size)  # 计算用于叠加的图片数
        img, label = data
        lam = np.random.beta(self.alpha, self.alpha)  # 融合因子

        # 原图和标签的 MixUp.
        img = lam * img + (1 - lam) * img[index, :]
        label = lam * batch_scores + (1 - lam) * batch_scores[index, :]
        # 由于此时返回的是 onehot 编码的 label，model 的 forward 也需要做相应调整
        return tuple(img, label)
```

因此，除了数据搬运和归一化，`data_preprocessor` 另一大功能就是数据批增强（`BatchAugmentation`）。数据预处理器的模块化也能帮助我们实现算法和数据增强之间的自由组合。

---

<kbd><b>Question</b></kbd>：如果 `DataLoader` 的输出和模型的输入类型不匹配怎么办，是修改 `DataLoader` 还是修改模型接口？

<kbd><b>Answer</b></kbd>：答案是都不合适。理想的解决方案是我们能够在不破坏模型和数据已有接口的情况下完成适配。这个时候数据预处理器也能承担类型转换的工作，例如将传入的 `data` 从 `tuple` 转换成指定字段的 `dict`。

---

看到这里，相信你已经能够理解数据预处理器存在的合理性，并且也能够自信地回答教程最初提出的两个问题！但是你可能还会疑惑 `train_step` 接口中传入的 `optim_wrapper` 又是什么，`test_step` 和 `val_step` 返回的结果和 `evaluator` 又有怎样的关系，这些问题会在模型精度评测教程和优化器封装得到解答。

# 4. 模型精度评测（EVALUATION）

在模型验证和模型测试中，通常需要对模型精度做定量评测。我们可以通过在配置文件中指定评测指标（Metric）来实现这一功能。

## 4.1 在模型训练或测试中进行评测

### 4.1.1 使用单个评测指标

在基于 MMEngine 进行模型训练或测试时，用户只需要在配置文件中通过 `val_evaluator` 和 `test_evaluator` 2 个字段分别指定模型验证和测试阶段的评测指标即可。例如，用户在使用 MMPretrain 训练分类模型时，希望在模型验证阶段评测 top-1 和 top-5 分类正确率，可以按以下方式配置：

```python
val_evaluator = dict(type='Accuracy', top_k=(1, 5))  # 使用分类正确率评测指标
```

> 关于具体评测指标的参数设置，用户可以查阅相关算法库的文档。如上例中的 [Accuracy 文档](https://mmpretrain.readthedocs.io/en/latest/api/generated/mmpretrain.evaluation.Accuracy.html#mmpretrain.evaluation.Accuracy)。

### 4.1.2 使用多个评测指标

如果需要同时评测多个指标，也可以将 `val_evaluator` 或 `test_evaluator` 设置为一个列表，其中每一项为一个评测指标的配置信息。例如，在使用 MMDetection 训练全景分割模型时，希望在模型测试阶段同时评测模型的目标检测（COCO AP/AR）和全景分割精度，可以按以下方式配置：

```python
test_evaluator = [
    # 目标检测指标
    dict(
        type='CocoMetric',
        metric=['bbox', 'segm'],
        ann_file='annotations/instances_val2017.json',
    ),
    # 全景分割指标
    dict(
        type='CocoPanopticMetric',
        ann_file='annotations/panoptic_val2017.json',
        seg_prefix='annotations/panoptic_val2017',
    )
]
```

### 4.1.3 自定义评测指标

如果算法库中提供的常用评测指标无法满足需求，用户也可以增加自定义的评测指标。我们以简化的分类正确率为例，介绍实现自定义评测指标的方法：

1. 在定义新的评测指标类时，需要继承基类 `BaseMetric`（关于该基类的介绍，可以参考[设计文档](https://mmengine.readthedocs.io/zh-cn/latest/design/evaluation.html)）。此外，评测指标类需要用注册器 `METRICS` 进行注册。

2. 实现 `process()` 方法。该方法有 2 个输入参数，我们从中分别取出样本类别标签（`gt`）和分类预测结果（`score`）--> `score, gt = data_samples`，并存放在 `self.results` 中。
   1. `data_batch`：一个Batch的测试数据样本
   2. `data_samples`：模型预测结果
   
3. 实现 `compute_metrics() 方法`。该方法有 1 个输入参数 `results`，里面存放了所有批次测试数据经过 `process()` 方法处理后得到的结果。从中取出样本类别标签和分类预测结果，即可计算得到分类正确率 `acc`。最终，将计算得到的评测指标以字典的形式返回。

4. （可选）可以为类属性 `default_prefix` 赋值。该属性会自动作为输出的评测指标名前缀（如 `defaut_prefix='my_metric'`，则实际输出的评测指标名为 `'my_metric/acc'`），用以进一步区分不同的评测指标。该前缀也可以在配置文件中通过 `prefix` 参数改写。我们建议在 `docstring` 中说明该评测指标类的 `default_prefix` 值以及所有的返回指标名称。

具体实现如下：

```python
from typing import Sequence, List

from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

import numpy as np


@METRICS.register_module()  # 将 Accuracy 类注册到 METRICS 注册器
class SimpleAccuracy(BaseMetric):
    """ Accuracy Evaluator

    Default prefix: ACC

    Metrics:
        - accuracy (float): classification accuracy
    """

    default_prefix = 'ACC'  # 设置 default_prefix

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]):
        """Process one batch of data and predictions. The processed
        Results should be stored in `self.results`, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """

        # 取出分类预测结果和类别标签
        result = {
            'pred': data_samples['pred_label'],
            'gt': data_samples['data_sample']['gt_label']
        }

        # 将当前 batch 的结果存进 self.results
        self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        # 汇总所有样本的分类预测结果和类别标签
        preds = np.concatenate([res['pred'] for res in results])
        gts = np.concatenate([res['gt'] for res in results])

        # 计算分类正确率
        acc = (preds == gts).sum() / preds.size

        # 返回评测指标结果
        return {'accuracy': acc}
```

## 4.2 使用离线结果进行评测

另一种常见的模型评测方式，是利用提前保存在文件中的模型预测结果进行离线评测。此时，用户需要手动构建评测器，并调用评测器的相应接口完成评测。

---

<kbd><b>Question</b></kbd>：对于模型指标评估而言，什么是在线评估，什么是离线评估？

<kbd><b>Answer</b></kbd>：在线评估和离线评估是模型指标评估的两种主要方式，它们在数据科学和机器学习领域中用于衡量模型性能。两者的主要区别在于评估过程中是否与实时数据流交互。

**离线评估**：
- 离线评估通常发生在模型开发阶段，使用已经收集好的、固定的数据集来评估模型的性能。
- 在离线评估中，模型的输出是已知的，因为使用的是历史数据，可以与实际结果进行对比。
- 离线评估允许研究者或数据科学家进行大量的实验，尝试不同的模型和参数，而不会对生产环境造成影响。
- 这种评估方式的缺点是，它可能无法完全反映模型在实际应用中的表现，因为现实世界的数据是动态变化的。

**在线评估**：
- 在线评估则是在模型部署后，使用实时数据流对模型性能进行监控和评估。
- 在线评估能够反映模型在实际应用中的即时表现，帮助及时发现模型性能的下降或者数据分布的变化。
- 这种评估方式需要考虑系统的性能和资源的限制，因为实时数据需要被快速处理并反馈结果。
- 在线评估有助于实现模型的持续学习和优化，但同时也需要有效的监控机制来确保模型的决策是安全和可靠的。

在实际应用中，通常会结合使用离线评估和在线评估，以确保模型的性能不仅在实验室条件下表现良好，而且能够在实际操作中持续满足业务需求。

---

关于离线评测的详细说明，以及评测器和评测指标的关系，可以参考设计文档。我们仅在此给出一个离线评测示例：

```python
from mmengine.evaluator import Evaluator
from mmengine.fileio import load


# 构建评测器。参数 `metrics` 为评测指标配置
evaluator = Evaluator(metrics=dict(type='Accuracy', top_k=(1, 5)))

# 从文件中读取测试数据。数据格式需要参考具体使用的 metric。
data = load('test_data.pkl')

# 从文件中读取模型预测结果。该结果由待评测算法在测试数据集上推理得到。
# 数据格式需要参考使用的 metric。
data_samples = load('prediction.pkl')

# 调用评测器离线评测接口，得到评测结果
# chunk_size 表示每次处理的样本数量，可根据内存大小调整
results = evaluator.offline_evaluate(data, data_samples, chunk_size=128)
```

# 5. 优化器封装（OPTIMWRAPPER）

在执行器教程和模型教程中，我们或多或少地提到了优化器封装（`OptimWrapper`）的概念，但是却没有介绍为什么我们需要优化器封装，相比于 PyTorch 原生的优化器，优化器封装又有怎样的优势，这些问题会在本教程中得到一一解答。我们将通过对比的方式帮助大家理解，优化器封装的优势，以及如何使用它。

优化器封装顾名思义，是 PyTorch 原生优化器（`Optimizer`）高级抽象，它在增加了更多功能的同时，提供了一套统一的接口。优化器封装支持不同的训练策略，包括**混合精度训练**、**梯度累加**和**梯度截断**。我们可以根据需求选择合适的训练策略。优化器封装还定义了一套标准的参数更新流程，用户可以基于这一套流程，实现同一套代码，不同训练策略的切换。

## 5.1 优化器封装 vs 优化器

这里我们分别基于 PyTorch 内置的优化器和 MMEngine 的优化器封装进行单精度训练、混合精度训练和梯度累加，对比二者实现上的区别。

### 5.1.1 训练模型

#### 5.1.1.1 基于 PyTorch 的 SGD 优化器实现单精度训练

```python
import torch
from torch.optim import SGD
import torch.nn as nn
import torch.nn.functional as F


inputs = [torch.zeros(10, 1, 1)] * 10
targets = [torch.ones(10, 1, 1)] * 10
model = nn.Linear(1, 1)
optimizer = SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()

for input, target in zip(inputs, targets):
    output = model(input)
    loss = F.l1_loss(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

#### 5.1.1.2 使用 MMEngine 的优化器封装实现单精度训练

```python
import torch
from torch.optim import SGD
import torch.nn as nn
import torch.nn.functional as F

from mmengine.optim import OptimWrapper


inputs = [torch.zeros(10, 1, 1)] * 10
targets = [torch.ones(10, 1, 1)] * 10
model = nn.Linear(1, 1)
optimizer = SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()

optim_wrapper = OptimWrapper(optimizer=optimizer)

for input, target in zip(inputs, targets):
    output = model(input)
    loss = F.l1_loss(output, target)
    optim_wrapper.update_params(loss)
```

优化器封装的 `update_params` 实现了标准的梯度计算、参数更新和梯度清零流程，可以直接用来更新模型参数。

#### 5.1.1.3 基于 PyTorch 的 SGD 优化器实现混合精度训练

```python
...

from torch.cuda.amp import autocast

for input, target in zip(inputs, targets):
    with autocast():
        output = model(input.cuda())
    loss = F.l1_loss(output, target.cuda())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

#### 5.1.1.4 基于 MMEngine 的 优化器封装实现混合精度训练

```python
...

from mmengine.optim import AmpOptimWrapper

# 从原来的OptimWrapper修改为AMPOptimWrapper
optim_wrapper = AmpOptimWrapper(optimizer=optimizer)

for input, target in zip(inputs, targets):
    with optim_wrapper.optim_context(model):
        output = model(input.cuda())
    loss = F.l1_loss(output, target.cuda())
    optim_wrapper.update_params(loss)
```

开启混合精度训练需要使用 `AmpOptimWrapper`，它的 `optim_context` 接口类似 `autocast`，会开启混合精度训练的上下文。除此之外它还能加速分布式训练时的梯度累加，这个我们会在下一个示例中介绍。

#### 5.1.1.5 基于 PyTorch 的 SGD 优化器实现混合精度训练和梯度累加

```python
for idx, (input, target) in enumerate(zip(inputs, targets)):
    with autocast():
        output = model(input.cuda())
    loss = F.l1_loss(output, target.cuda())
    loss.backward()
    if idx % 2 == 0:  # 每两次迭代更新一次梯度
        optimizer.step()
        optimizer.zero_grad()
```

#### 5.1.1.6 基于 MMEngine 的优化器封装实现混合精度训练和梯度累加

```python
# 注意accumulative_counts参数
optim_wrapper = AmpOptimWrapper(optimizer=optimizer, accumulative_counts=2)

# 下面的代码与AMP代码是一致的，但optim_wrapper中accumulative_counts=2
for input, target in zip(inputs, targets):
    with optim_wrapper.optim_context(model):
        output = model(input.cuda())
    loss = F.l1_loss(output, target.cuda())
    optim_wrapper.update_params(loss)
```

我们只需要配置 `accumulative_counts` 参数，并调用 `update_params` 接口就能实现梯度累加的功能。除此之外，分布式训练情况下，如果我们配置梯度累加的同时开启了 `optim_wrapper` 上下文，可以避免梯度累加阶段不必要的梯度同步。

> 🤓  OpenMMLab 这个项目感觉就不是一个深度学习的库，不像 Ultralytics 团队那样，尽可能使用 PyTorch 官方的方法，而是对 PyTorch 进行了二次封装，这导致学习 OpenMMLab 跟学习 PyTorch 的成本差不多，也容易与 PyTorch 语法混淆。

优化器封装同样提供了更细粒度的接口，方便用户实现一些自定义的参数更新逻辑：

- `backward`：传入损失，用于计算参数梯度。
- `step`：同 `optimizer.step`，用于更新参数。
- `zero_grad`：同 `optimizer.zero_grad`，用于参数的梯度。

我们可以使用上述接口实现和 PyTorch 优化器相同的参数更新逻辑：

```python
for idx, (input, target) in enumerate(zip(inputs, targets)):
    optimizer.zero_grad()
    with optim_wrapper.optim_context(model):  # 这里不是autocast
        output = model(input.cuda())
    loss = F.l1_loss(output, target.cuda())
    optim_wrapper.backward(loss)
    if idx % 2 == 0:
        optim_wrapper.step()
        optim_wrapper.zero_grad()
```

我们同样可以为优化器封装配置梯度裁减策略：

```python
# 基于 torch.nn.utils.clip_grad_norm_ 对梯度进行裁减
optim_wrapper = AmpOptimWrapper(optimizer=optimizer, 
                                clip_grad=dict(max_norm=1))

# 基于 torch.nn.utils.clip_grad_value_ 对梯度进行裁减
optim_wrapper = AmpOptimWrapper(optimizer=optimizer, 
                                clip_grad=dict(clip_value=0.2))
```

### 5.1.2 获取学习率/动量

优化器封装提供了 `get_lr` 和 `get_momentum` 接口用于获取优化器的一个参数组的学习率：

```python
import torch.nn as nn
from torch.optim import SGD

from mmengine.optim import OptimWrapper


model = nn.Linear(1, 1)

# Step 1: 先创建一个PyTorch的优化器
optimizer = SGD(model.parameters(), lr=0.01)  # PyTorch的优化器定义方法

# Step2: 再使用MMEngine的OptimWrapper进行封装
optim_wrapper = OptimWrapper(optimizer)  # MMEngine的优化器定义方法

# PyTorch获取学习率的动量的方法
print(optimizer.param_groups[0]['lr'])  # 0.01
print(optimizer.param_groups[0]['momentum'])  # 0

# MMEngine获取学习率的动量的方法
print(optim_wrapper.get_lr())  # {'lr': [0.01]}
print(optim_wrapper.get_momentum())  # {'momentum': [0]}

# MMEngine取出数值
print(optim_wrapper.get_lr()['lr'][0])  # 0.01
print(optim_wrapper.get_momentum()['momentum'][0])  # 0
```

### 5.1.3 导出/加载状态字典

优化器封装和优化器一样，提供了 `state_dict` 和 `load_state_dict` 接口，用于导出/加载优化器状态，对于 `AmpOptimWrapper`，优化器封装还会额外导出混合精度训练相关的参数：

```python
import torch.nn as nn
from torch.optim import SGD
from mmengine.optim import OptimWrapper, AmpOptimWrapper


model = nn.Linear(1, 1)
optimizer = SGD(model.parameters(), lr=0.01)

optim_wrapper = OptimWrapper(optimizer=optimizer)  # MMEngine普通的优化器
amp_optim_wrapper = AmpOptimWrapper(optimizer=optimizer)  # MMEngine使用AMP的优化器

# 导出状态字典
optim_state_dict = optim_wrapper.state_dict()
amp_optim_state_dict = amp_optim_wrapper.state_dict()

print(f"optim_state_dict: \n{optim_state_dict}\n")
print(f"amp_optim_state_dict: \n{amp_optim_state_dict}\n")

optim_wrapper_new = OptimWrapper(optimizer=optimizer)
amp_optim_wrapper_new = AmpOptimWrapper(optimizer=optimizer)

# 加载状态字典
amp_optim_wrapper_new.load_state_dict(amp_optim_state_dict)
optim_wrapper_new.load_state_dict(optim_state_dict)
print("状态字典加载完毕!")
```

```
optim_state_dict: 
{'state': {}, 'param_groups': [{'lr': 0.01, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'foreach': None, 'params': [0, 1]}]}

amp_optim_state_dict: 
{'state': {}, 'param_groups': [{'lr': 0.01, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'foreach': None, 'params': [0, 1]}], 'loss_scaler': {'scale': 65536.0, 'growth_factor': 2.0, 'backoff_factor': 0.5, 'growth_interval': 2000, '_growth_tracker': 0}}

状态字典加载完毕!
```

### 5.1.4 使用多个优化器

考虑到生成对抗网络之类的算法通常需要使用多个优化器来训练生成器和判别器，因此优化器封装提供了优化器封装的容器类：`OptimWrapperDict` 来管理多个优化器封装。`OptimWrapperDict` 以字典的形式存储优化器封装，并允许用户像字典一样访问、遍历其中的元素，即优化器封装实例。

与普通的优化器封装不同，`OptimWrapperDict` 没有实现` update_params`、 `optim_context`, `backward、step` 等方法，无法被直接用于训练模型。我们建议直接访问 `OptimWrapperDict` 管理的优化器实例，来实现参数更新逻辑。

你或许会好奇，既然 `OptimWrapperDict` 没有训练的功能，那为什么不直接使用 `dict` 来管理多个优化器？事实上，`OptimWrapperDict` 的核心功能是支持批量导出/加载所有优化器封装的状态字典；支持获取多个优化器封装的学习率、动量。如果没有 `OptimWrapperDict`，`MMEngine` 就需要在很多位置对优化器封装的类型做 `if else` 判断，以获取所有优化器封装的状态。

```python
import torch.nn as nn
from torch.optim import SGD
from mmengine.optim import OptimWrapper, OptimWrapperDict


# 创建模型
gen = nn.Linear(1, 1)  # 生成网络
disc = nn.Linear(1, 1)  # 判别网络

# 创建PyTorch优化器
optimizer_gen = SGD(gen.parameters(), lr=0.01)
optimizer_disc = SGD(disc.parameters(), lr=0.01)

# 创建MMEngine优化器
optim_wapper_gen = OptimWrapper(optimizer=optimizer_gen)
optim_wapper_disc = OptimWrapper(optimizer=optimizer_disc)

# 优化器字典
optim_dict = OptimWrapperDict(gen=optim_wapper_gen, 
                              disc=optim_wapper_disc)

# 获取MMEngine优化器字典中的所有学习率和动量
print(f"MMEngine优化器字典中的所有学习率：{optim_dict.get_lr()}")  # {'gen.lr': [0.01], 'disc.lr': [0.01]}
print(f"MMEngine优化器字典中的所有动量：{optim_dict.get_momentum()}")  # {'gen.lr': [0.01], 'disc.lr': [0.01]}
```

```
MMEngine优化器字典中的所有学习率：{'gen.lr': [0.01], 'disc.lr': [0.01]}
MMEngine优化器字典中的所有动量：{'gen.momentum': [0], 'disc.momentum': [0]}
```

如上例所示，`OptimWrapperDict` 可以非常方便的导出所有优化器封装的学习率和动量，同样的，优化器封装也能够导出/加载所有优化器封装的状态字典。

### 5.1.5 在执行器（Runner）中配置优化器封装

优化器封装需要接受 `optimizer` 参数，因此我们首先需要为优化器封装配置 `optimizer`。MMEngine 会自动将 PyTorch 中的所有优化器都添加进 `OPTIMIZERS` 注册表中，用户可以用字典的形式来指定优化器，所有支持的优化器见 [PyTorch 优化器列表](https://pytorch.org/docs/stable/optim.html#algorithms)。

以配置一个 SGD 优化器封装为例：

```python
# Step 1: 先创建一个PyTorch的优化器
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# Step2: 再使用MMEngine的OptimWrapper进行封装
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
```

这样我们就配置好了一个优化器类型为 SGD 的优化器封装，学习率、动量等参数如配置所示。考虑到 `OptimWrapper` 为标准的单精度训练，因此我们也可以不配置 `type` 字段：

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(optimizer=optimizer)
```

要想开启混合精度训练和梯度累加，需要将 `type` 切换成 `AmpOptimWrapper`，并指定 `accumulative_counts` 参数：

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer, accumulative_counts=2)
```

## 5.2 进阶配置

PyTorch 的优化器支持对模型中的不同参数设置不同的超参数，例如对一个分类模型的骨干（backbone）和分类头（head）设置不同的学习率：

```python
from torch.optim import SGD
import torch.nn as nn


model = nn.ModuleDict(dict(backbone=nn.Linear(1, 1), head=nn.Linear(1, 1)))
optimizer = SGD(
    [
        {'params': model.backbone.parameters()},  # 没有指定学习率则使用后面公用的学习率
        {'params': model.head.parameters(), 'lr': 0.003}  # 指定了学习率，则使用自己指定的学习率
    ],
    lr=0.01,  # 公用的学习率
    momentum=0.9  # 公用的动量
)

for value in optimizer.param_groups:
    print(f"{value}\n")
```

```
{'params': [Parameter containing:
tensor([[-0.6182]], requires_grad=True), Parameter containing:
tensor([0.4877], requires_grad=True)], 'lr': 0.01, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'foreach': None, 'differentiable': False}

{'params': [Parameter containing:
tensor([[-0.1764]], requires_grad=True), Parameter containing:
tensor([0.7182], requires_grad=True)], 'lr': 0.003, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'foreach': None, 'differentiable': False}
```

上面的例子中，模型的骨干部分使用了 `0.01` 学习率，而模型的头部则使用了 `0.003` 学习率。用户可以将模型的不同部分参数和对应的超参组成一个字典的列表传给优化器，来实现对模型优化的细粒度调整。

在 MMEngine 中，我们通过优化器封装构造器（optimizer wrapper constructor），让用户能够直接通过设置优化器封装配置文件中的 `paramwise_cfg` 字段而非修改代码来实现对模型的不同部分设置不同的超参。

### 5.2.1 为不同类型的参数设置不同的超参系数

MMEngine 提供的默认优化器封装构造器支持对模型中不同类型的参数设置不同的超参系数。例如，我们可以在 `paramwise_cfg` 中设置 `norm_decay_mult=0`，从而将正则化层（normalization layer）的权重（weight）和偏置（bias）的权值衰减系数（weight decay）设置为 0，来实现 Bag of Tricks 论文中提到的不对正则化层进行权值衰减的技巧。

具体示例如下，我们将 ToyModel 中所有正则化层（`head.bn`）的权重衰减系数设置为 0：

```python
import torch.nn as nn
from mmengine.optim import build_optim_wrapper
from collections import OrderedDict


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.ModuleDict(
            dict(
                layer0=nn.Linear(1, 1),
                layer1=nn.Linear(1, 1)
            )
        )

        self.head = nn.Sequential(
            OrderedDict(
                linear=nn.Linear(1, 1),
                bn=nn.BatchNorm1d(1)
            )
        )


optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.01,
        weight_decay=0.0001),
    paramwise_cfg=dict(norm_decay_mult=0)
)

optimizer = build_optim_wrapper(ToyModel(), optim_wrapper)
```

```
02/28 10:37:42 - mmengine - INFO - paramwise_options -- head.bn.weight:weight_decay=0.0
02/28 10:37:42 - mmengine - INFO - paramwise_options -- head.bn.bias:weight_decay=0.0
```

除了可以对正则化层的权重衰减进行配置外，MMEngine 的默认优化器封装构造器的 `paramwise_cfg` 还支持对更多不同类型的参数设置超参系数，支持的配置如下：

|param name|description|
|:-|:-|
|`lr_mult`|所有参数的学习率系数|
|`decay_mult`|所有参数的衰减系数|
|`bias_lr_mult`|偏置的学习率系数（不包括正则化层的偏置以及可变形卷积的 offset）|
|`bias_decay_mult`|偏置的权值衰减系数（不包括正则化层的偏置以及可变形卷积的 offset）|
|`norm_decay_mult`|正则化层权重和偏置的权值衰减系数|
|`flat_decay_mult`|一维参数的权值衰减系数|
|`dwconv_decay_mult`|Depth-wise 卷积的权值衰减系数|
|`bypass_duplicate`|是否跳过重复的参数，默认为 `False`|
|`dcn_offset_lr_mult`|可变形卷积（Deformable Convolution）的学习率系数|

### 5.2.2 为模型不同部分的参数设置不同的超参系数

此外，与上文 PyTorch 的示例一样，在 MMEngine 中我们也同样可以对模型中的任意模块设置不同的超参，只需要在 `paramwise_cfg` 中设置 `custom_keys` 即可。

例如我们想将 `backbone.layer0` 所有参数的学习率设置为 0，衰减系数设置为 0，backbone 其余子模块的学习率设置为 0.01；head 所有参数的学习率设置为 0.001，可以这样配置：

```python
import torch.nn as nn
from collections import OrderedDict
from mmengine.optim import build_optim_wrapper


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.ModuleDict(
            dict(
                layer0=nn.Linear(1, 1),
                layer1=nn.Linear(1, 1)
            )
        )

        self.head = nn.Sequential(
            OrderedDict(
                linear=nn.Linear(1, 1),
                bn=nn.BatchNorm1d(1)
            )
        )


optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.layer0': dict(lr_mult=0,  # 所有参数的学习率系数
                                    decay_mult=0),  # 所有参数的衰减系数
            'backbone': dict(lr_mult=1),
            'head': dict(lr_mult=0.1)
        }))

optimizer = build_optim_wrapper(ToyModel(), optim_wrapper)
```

```
02/28 10:43:26 - mmengine - INFO - paramwise_options -- backbone.layer0.weight:lr=0.0
02/28 10:43:26 - mmengine - INFO - paramwise_options -- backbone.layer0.weight:weight_decay=0.0
02/28 10:43:26 - mmengine - INFO - paramwise_options -- backbone.layer0.weight:lr_mult=0
02/28 10:43:26 - mmengine - INFO - paramwise_options -- backbone.layer0.weight:decay_mult=0
02/28 10:43:26 - mmengine - INFO - paramwise_options -- backbone.layer0.bias:lr=0.0
02/28 10:43:26 - mmengine - INFO - paramwise_options -- backbone.layer0.bias:weight_decay=0.0
02/28 10:43:26 - mmengine - INFO - paramwise_options -- backbone.layer0.bias:lr_mult=0
02/28 10:43:26 - mmengine - INFO - paramwise_options -- backbone.layer0.bias:decay_mult=0
02/28 10:43:26 - mmengine - INFO - paramwise_options -- backbone.layer1.weight:lr=0.01
02/28 10:43:26 - mmengine - INFO - paramwise_options -- backbone.layer1.weight:weight_decay=0.0001
02/28 10:43:26 - mmengine - INFO - paramwise_options -- backbone.layer1.weight:lr_mult=1
02/28 10:43:26 - mmengine - INFO - paramwise_options -- backbone.layer1.bias:lr=0.01
02/28 10:43:26 - mmengine - INFO - paramwise_options -- backbone.layer1.bias:weight_decay=0.0001
02/28 10:43:26 - mmengine - INFO - paramwise_options -- backbone.layer1.bias:lr_mult=1
02/28 10:43:26 - mmengine - INFO - paramwise_options -- head.linear.weight:lr=0.001
02/28 10:43:26 - mmengine - INFO - paramwise_options -- head.linear.weight:weight_decay=0.0001
02/28 10:43:26 - mmengine - INFO - paramwise_options -- head.linear.weight:lr_mult=0.1
02/28 10:43:26 - mmengine - INFO - paramwise_options -- head.linear.bias:lr=0.001
02/28 10:43:26 - mmengine - INFO - paramwise_options -- head.linear.bias:weight_decay=0.0001
02/28 10:43:26 - mmengine - INFO - paramwise_options -- head.linear.bias:lr_mult=0.1
02/28 10:43:26 - mmengine - INFO - paramwise_options -- head.bn.weight:lr=0.001
02/28 10:43:26 - mmengine - INFO - paramwise_options -- head.bn.weight:weight_decay=0.0001
02/28 10:43:26 - mmengine - INFO - paramwise_options -- head.bn.weight:lr_mult=0.1
02/28 10:43:26 - mmengine - INFO - paramwise_options -- head.bn.bias:lr=0.001
02/28 10:43:26 - mmengine - INFO - paramwise_options -- head.bn.bias:weight_decay=0.0001
02/28 10:43:26 - mmengine - INFO - paramwise_options -- head.bn.bias:lr_mult=0.1
```

`custom_keys` 中每一个字段的含义如下：

- `'backbone': dict(lr_mult=1)`：将名字前缀为 `backbone` 的参数的学习率系数设置为 1
- `'backbone.layer0': dict(lr_mult=0, decay_mult=0)`：将名字前缀为 `backbone.layer0` 的参数学习率系数设置为 0，衰减系数设置为 0，该配置优先级比第一条高
- `'head': dict(lr_mult=0.1)`：将名字前缀为 `head` 的参数的学习率系数设置为 0.1

---

上例中，模型的状态字典的 `key` 如下：

```python
for name, value in ToyModel().named_parameters():
    print(name)
```

```
backbone.layer0.weight
backbone.layer0.bias
backbone.layer1.weight
backbone.layer1.bias
head.linear.weight
head.linear.bias
head.bn.weight
head.bn.bias
```

### 5.2.3 自定义优化器构造策略

与 MMEngine 中的其他模块一样，优化器封装构造器也同样由注册表（Registry）管理。我们可以通过实现自定义的优化器封装构造器来实现自定义的超参设置策略。

例如，我们想实现一个叫做 `LayerDecayOptimWrapperConstructor` 的优化器封装构造器，能够对模型不同深度的层自动设置递减的学习率：

```python
import torch.nn as nn
from mmengine.optim import build_optim_wrapper, DefaultOptimWrapperConstructor
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.logging import print_log


@OPTIM_WRAPPER_CONSTRUCTORS.register_module(force=True)
class LayerDecayOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        super().__init__(optim_wrapper_cfg, paramwise_cfg=None)
        self.decay_factor = paramwise_cfg.get('decay_factor', 0.5)

        super().__init__(optim_wrapper_cfg, paramwise_cfg)

    def add_params(self, params, module, prefix='', lr=None):
        if lr is None:
            lr = self.base_lr

        for name, param in module.named_parameters(recurse=False):
            param_group = dict()
            param_group['params'] = [param]
            param_group['lr'] = lr
            params.append(param_group)
            full_name = f'{prefix}.{name}' if prefix else name
            print_log(f'{full_name} : lr={lr}', logger='current')

        for name, module in module.named_children():
            chiled_prefix = f'{prefix}.{name}' if prefix else name
            
            self.add_params(params, module, 
                            chiled_prefix, 
                            lr=lr * self.decay_factor)


class ToyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.ModuleDict(dict(linear=nn.Linear(1, 1)))
        self.linear = nn.Linear(1, 1)


model = ToyModel()

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
    paramwise_cfg=dict(decay_factor=0.5),
    constructor='LayerDecayOptimWrapperConstructor')

optimizer = build_optim_wrapper(model, optim_wrapper)
```

```
02/28 11:06:40 - mmengine - INFO - layer.linear.weight : lr=0.0025
02/28 11:06:40 - mmengine - INFO - layer.linear.bias : lr=0.0025
02/28 11:06:40 - mmengine - INFO - linear.weight : lr=0.005
02/28 11:06:40 - mmengine - INFO - linear.bias : lr=0.005
```

`add_params` 被第一次调用时，`params` 参数为空列表（`list`），`module` 为模型（model）。详细的重载规则参考[优化器封装构造器文档](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.optim.DefaultOptimWrapperConstructor.html#mmengine.optim.DefaultOptimWrapperConstructor)。

类似地，如果想构造多个优化器，也需要实现自定义的构造器：

```python
@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MultipleOptimiWrapperConstructor:
    ...
```

### 5.2.4 在训练过程中调整超参

优化器中的超参数在构造时只能设置为一个定值，仅仅使用优化器封装，并不能在训练过程中调整学习率等参数。在 MMEngine 中，我们实现了参数调度器（Parameter Scheduler），以便能够在训练过程中调整参数。

# 6. 优化器参数调整策略（PARAMETER SCHEDULER）

在模型训练过程中，我们往往不是采用固定的优化参数，例如学习率等，会随着训练轮数的增加进行调整。最简单常见的学习率调整策略就是阶梯式下降，例如每隔一段时间将学习率降低为原来的几分之一。PyTorch 中有学习率调度器 `LRScheduler` 来对各种不同的学习率调整方式进行抽象，但支持仍然比较有限，在 MMEngine 中，我们对其进行了拓展，实现了更通用的参数调度器，可以对学习率、动量等优化器相关的参数进行调整，并且支持多个调度器进行组合，应用更复杂的调度策略。

## 6.1 参数调度器的使用

我们先简单介绍一下如何使用 PyTorch 内置的学习率调度器来进行学习率的调整。下面是参考 PyTorch 官方文档 实现的一个例子，我们构造一个 `ExponentialLR`，并且在每个 `epoch` 结束后调用 `scheduler.step()`，实现了随 `epoch` 指数下降的学习率调整策略。

```python
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR


model = torch.nn.Linear(1, 1)
dataset = [torch.randn((1, 1, 1)) for _ in range(20)]
optimizer = SGD(model, 0.1)
scheduler = ExponentialLR(optimizer, gamma=0.9)

for epoch in range(10):  # epoch
    for data in dataset:  # iteration
        optimizer.zero_grad()  # 优化器梯度清零
        output = model(data)  # 前向推理
        loss = 1 - output  # 计算损失
        loss.backward()  # 损失反向传播到优化器
        optimizer.step()  # 优化器执行参数更新
    scheduler.step()  # 调度器随着epoch的进行而执行
```

在 `mmengine.optim.scheduler` 中，我们支持大部分 PyTorch 中的学习率调度器，例如 `ExponentialLR`，`LinearLR`，`StepLR`，`MultiStepLR` 等，使用方式也基本一致，所有支持的调度器见调度器接口文档。同时增加了对动量的调整，在类名中将 `LR` 替换成 `Momentum` 即可，例如 `ExponentialMomentum`，`LinearMomentum`。更进一步地，我们实现了通用的参数调度器 `ParamScheduler`，用于调整优化器的中的其他参数，包括 `weight_decay` 等。这个特性可以很方便地配置一些新算法中复杂的调整策略。

### 6.1.1 使用单一的学习率调度器

如果整个训练过程只需要使用一个学习率调度器, 那么和 PyTorch 自带的学习率调度器没有差异。

```python
import torch
from torch.optim import SGD
from mmengine.runner import Runner
from mmengine.optim.scheduler import MultiStepLR


model = torch.nn.Linear(1, 1)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
param_scheduler = MultiStepLR(optimizer, milestones=[8, 11], gamma=0.1)

runner = Runner(
    model=model,
    optim_wrapper=dict(optimizer=optimizer),
    param_scheduler=param_scheduler,
)
```

<div align=center>
    <img src=./imgs_markdown/2024-02-28-11-19-13.png
    width=50%>
    <center></center>
</div>

如果配合注册器（Registry）和配置（Config）文件使用的话，我们可以设置配置文件中的 `param_scheduler` 字段来指定调度器，执行器（Runner）会根据此字段以及执行器中的优化器自动构建学习率调度器：

```python
# 在配置文件中设置学习率调度器字段
param_scheduler = dict(type='MultiStepLR', 
                       by_epoch=True, 
                       milestones=[8, 11], 
                       gamma=0.1)
```

注意这里增加了初始化参数 `by_epoch`，控制的是学习率调整频率，当其为 `True` 时表示按轮次（epoch）调整，为 `False` 时表示按迭代次数（iteration）调整，默认值为 `True`。在上面的例子中，表示按照轮次进行调整，此时其他参数的单位均为 epoch，例如 `milestones` 中的 `[8, 11]` 表示第 8 和 11 个 epoch 结束时，学习率将会被调整为上一轮次的 0.1 倍。

当修改了学习率调整频率后，调度器中与计数相关设置的含义也会相应被改变。当 `by_epoch=True` 时，`milestones` 中的数字表示在哪些轮次进行学习率衰减，而当 `by_epoch=False` 时则表示在进行到第几次迭代时进行学习率衰减。下面是一个按照迭代次数进行调整的例子，在第 600 和 800 次迭代结束时，学习率将会被调整为原来的 0.1 倍。

```python
param_scheduler = dict(type='MultiStepLR', 
                       by_epoch=False, 
                       milestones=[600, 800], 
                       gamma=0.1)
```

<div align=center>
    <img src=./imgs_markdown/2024-02-28-11-22-01.png
    width=50%>
    <center></center>
</div>

若用户希望在配置调度器（Scheduler）时按 Epoch 填写参数，同时可以使用基于迭代的更新频率，MMEngine 的调度器也提供了自动换算的方式。用户可以调用 `build_iter_from_epoch` 方法，并提供每个训练 `Epoch` 的迭代次数，即可构造按迭代次数更新的调度器对象：

```python
epoch_length = len(train_dataloader)
param_scheduler = MultiStepLR.build_iter_from_epoch(optimizer, 
                                                    milestones=[8, 11], 
                                                    gamma=0.1, 
                                                    epoch_length=epoch_length)
```

如果使用配置文件构建调度器，只需要在配置中加入 `convert_to_iter_based=True`，执行器会自动调用 `build_iter_from_epoch` 将基于 epoch 的配置文件转换为基于 iteration 次数的调度器对象：

```python
param_scheduler = dict(type='MultiStepLR', 
                       by_epoch=True, 
                       milestones=[8, 11], 
                       gamma=0.1, 
                       convert_to_iter_based=True)
```

为了能直观感受这两种模式的区别，我们这里再举一个例子。下面是一个按轮次更新的余弦退火（CosineAnnealing）学习率调度器，学习率仅在每个轮次结束后被修改：

```python
param_scheduler = dict(type='CosineAnnealingLR', 
                       by_epoch=True, 
                       T_max=12)
```

<div align=center>
    <img src=./imgs_markdown/2024-02-28-11-25-57.png
    width=50%>
    <center></center>
</div>

而在使用自动换算后，学习率会在每次迭代后被修改。从下图可以看出，学习率的变化更为平滑。

```python
param_scheduler = dict(type='CosineAnnealingLR', 
                       by_epoch=True, 
                       T_max=12, 
                       convert_to_iter_based=True)
```

<div align=center>
    <img src=./imgs_markdown/2024-02-28-11-26-42.png
    width=50%>
    <center></center>
</div>

### 6.1.2 组合多个学习率调度器（以学习率预热为例）

有些算法在训练过程中，并不是自始至终按照某个调度策略进行学习率调整的。最常见的例子是学习率预热（Warm-up），比如在训练刚开始的若干迭代次数使用线性的调整策略将学习率从一个较小的值增长到正常，然后按照另外的调整策略进行正常训练。

MMEngine 支持组合多个调度器一起使用，只需将配置文件中的 `scheduler` 字段修改为一组调度器配置的列表，`SchedulerStepHook` 可以自动对调度器列表进行处理。下面的例子便实现了学习率预热。

```python
param_scheduler = [
    # 线性学习率预热调度器
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=False,  # 按迭代更新学习率
         begin=0,
         end=50),  # 预热前 50 次迭代

    # 主学习率调度器
    dict(type='MultiStepLR',
         by_epoch=True,  # 按轮次更新学习率
         milestones=[8, 11],
         gamma=0.1)
]
```

<div align=center>
    <img src=./imgs_markdown/2024-02-28-11-28-23.png
    width=50%>
    <center></center>
</div>

注意这里增加了 `begin` 和 `end` 参数，这两个参数指定了调度器的生效区间。生效区间通常只在多个调度器组合时才需要去设置，使用单个调度器时可以忽略。当指定了 `begin` 和 `end` 参数时，表示该调度器只在 `[begin, end)` 区间内生效，其单位是由 `by_epoch` 参数决定。上述例子中预热阶段 `LinearLR` 的 `by_epoch` 为 `False`，表示该调度器只在前 50 次迭代生效，超过 50 次迭代后此调度器不再生效，由第二个调度器来控制学习率，即 `MultiStepLR`。在组合不同调度器时，各调度器的 `by_epoch` 参数不必相同。

这里再举一个例子：

```python
param_scheduler = [
    # 在 [0, 100) 迭代时使用线性学习率
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=100),

    # 在 [100, 900) 迭代时使用余弦学习率
    dict(type='CosineAnnealingLR',
         T_max=800,
         by_epoch=False,
         begin=100,
         end=900)
]
```

<div align=center>
    <img src=./imgs_markdown/2024-02-28-11-29-56.png
    width=50%>
    <center></center>
</div>

上述例子表示在训练的前 100 次迭代时使用线性的学习率预热，然后在第 100 到第 900 次迭代时使用周期为 800 的余弦退火学习率调度器使学习率按照余弦函数逐渐下降为 0 。

---

我们可以组合任意多个调度器，既可以使用 MMEngine 中已经支持的调度器，也可以实现自定义的调度器。💡  注意：

- 如果相邻两个调度器的生效区间没有紧邻，而是有一段区间没有被覆盖，那么这段区间的学习率维持不变。
- 如果两个调度器的生效区间发生了重叠，则对多组调度器叠加使用，学习率的调整会按照调度器配置文件中的顺序触发（行为与 PyTorch 中 ChainedScheduler 一致）。 

在一般情况下，我们推荐用户在训练的不同阶段使用不同的学习率调度策略来避免调度器的生效区间发生重叠。如果确实需要将两个调度器叠加使用，则需要十分小心，避免学习率的调整与预期不符。

## 6.2 如何调整其他参数

### 6.2.1 动量

和学习率一样，动量也是优化器参数组中一组可以调度的参数。动量调度器（`momentum scheduler`）的使用方法和学习率调度器完全一样。同样也只需要将动量调度器的配置添加进配置文件中的 `param_scheduler` 字段的列表中即可。

示例:

```python
param_scheduler = [
    # 线性学习率
    dict(type='LinearLR', ...),

    # 动量调度器
    dict(type='LinearMomentum',
         start_factor=0.001,
         by_epoch=False,  # 随着iter改变
         begin=0,  # 开始起作用的iter
         end=1000  # 结束的iter
    ) 
]
```

### 6.2.2 通用的参数调度器

MMEngine 还提供了一组通用的参数调度器用于调度优化器的 `param_groups` 中的其他参数，将学习率调度器类名中的 `LR` 改为 `Param` 即可，例如 `LinearParamScheduler`。用户可以通过设置参数调度器的 `param_name` 变量来选择想要调度的参数。

下面是一个通过自定义参数名来调度的例子：

```python
param_scheduler = [
    dict(type='LinearParamScheduler',
         param_name='lr',  # 调度 `optimizer.param_groups` 中名为 'lr' 的变量
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=1000)
]
```

这里设置的参数名是 `lr`，因此这个调度器的作用等同于直接使用学习率调度器 `LinearLRScheduler`。

除了动量之外，用户也可以对 `optimizer.param_groups` 中的其他参数名进行调度，可调度的参数取决于所使用的优化器。例如，当使用带 `weight_decay` 的 SGD 优化器时，可以按照以下示例对调整 `weight_decay`：

```python
param_scheduler = [
    dict(type='LinearParamScheduler',
         param_name='weight_decay',  # 调度 `optimizer.param_groups` 中名为 'weight_decay' 的变量
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=1000)
]
```

# 7. 钩子（HOOK）

钩子编程是一种编程模式，是指在程序的一个或者多个位置设置位点（挂载点），当程序运行至某个位点时，会自动调用运行时注册到位点的所有方法。钩子编程可以提高程序的灵活性和拓展性，用户将自定义的方法注册到位点便可被调用而无需修改程序中的代码。

MMEngine 提供了很多内置的钩子，将钩子分为两类，分别是

1. **内置的默认钩子**：会默认往执行器注册
2. **内置的自定义钩子**：需要用户自己注册

⚠️  每个钩子都有对应的优先级，在同一位点，钩子的优先级越高，越早被执行器调用，如果优先级一样，被调用的顺序和钩子注册的顺序一致。优先级列表如下（<u>数字越小优先级越高</u>）：

|Priority Name|Priority|
|:-|:-:|
|HIGHEST|(0)|
|VERY_HIGH|(10)|
|HIGH|(30)|
|ABOVE_NORMAL|(40)|
|NORMAL|(50)|
|BELOW_NORMAL|(60)|
|LOW|(70)|
|VERY_LOW|(90)|
|LOWEST|(100)|

## 7.1 内置的默认钩子

|Hook Name|Purpose|Priority</br>(High -> Low)|
|:-|:-|:-:|
|RuntimeInfoHook|往 message hub 更新运行时信息|VERY_HIGH (10)|
|IterTimerHook|统计迭代耗时|NORMAL (50)|
|DistSamplerSeedHook|确保分布式 Sampler 的 shuffle 生效|NORMAL (50)|
|LoggerHook|打印日志|BELOW_NORMAL (60)|
|ParamSchedulerHook|调用 ParamScheduler 的 step 方法|LOW (70)|
|CheckpointHook|按指定间隔保存权重|VERY_LOW (90)|

> ⚠️  不建议修改默认钩子的优先级，因为优先级低的钩子可能会依赖优先级高的钩子。例如 `CheckpointHook` 的优先级需要比 `ParamSchedulerHook` 低，这样保存的优化器状态才是正确的状态。

## 7.2 内置的自定义钩子

|Hook Name|Purpose|Priority</br>(High -> Low)|
|:-|:-|:-:|
|EMAHook|模型参数指数滑动平均|NORMAL (50)|
|EmptyCacheHook|PyTorch CUDA 缓存清理|NORMAL (50)|
|SyncBuffersHook|同步模型的 buffer|NORMAL (50)|
|ProfilerHook|分析算子的执行时间以及显存占用情况|VERY_LOW (90)|

> ⚠️  自定义钩子的优先级默认为 `NORMAL (50)`

两种钩子在执行器中的设置不同，默认钩子的配置传给执行器的 `default_hooks` 参数，自定义钩子的配置传给 `custom_hooks` 参数，如下所示：

```python
from mmengine.runner import Runner

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    logger=dict(type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
)

custom_hooks = [
    dict(
        type='EmptyCacheHook'
    )
]

runner = Runner(default_hooks=default_hooks, 
                custom_hooks=custom_hooks, 
                ...)
runner.train()
```

## 7.3 内置的默认钩子的用法

### 7.3.1 CheckpointHook

`CheckpointHook` 按照给定间隔保存模型的权重，如果是分布式多卡训练，则只有主（master）进程会保存权重。`CheckpointHook` 的主要功能如下：

1. 按照间隔保存权重，支持按 `epoch` 数或者 `iteration` 数保存权重
2. 保存最新的多个权重
3. 保存最优权重
4. 指定保存权重的路径
5. 制作发布用的权重
6. 设置开始保存权重的 `epoch` 数或者 `iteration` 数

如需了解其他功能，请阅读 [CheckpointHook API 文档](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.hooks.CheckpointHook.html#mmengine.hooks.CheckpointHook)。

---

下面介绍上面提到的 6 个功能。

〔**1. 按照间隔保存权重，支持按 epoch 数或者 iteration 数保存权重**〕

假设我们一共训练 20 个 epoch 并希望每隔 5 个 epoch 保存一次权重，下面的配置（Config）即可帮我们实现该需求。

```python
# by_epoch 的默认值为 True
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=5, 
        by_epoch=True
    )
)
```

如果想以迭代次数作为保存间隔，则可以将 `by_epoch` 设为 `False`，`interval=5` 则表示每迭代 5 次保存一次权重。

```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=5, 
        by_epoch=False
    )
)
```

〔**2. 保存最新的多个权重**〕

如果只想保存一定数量的权重，可以通过设置 `max_keep_ckpts` 参数实现最多保存 `max_keep_ckpts` 个权重，当保存的权重数超过 `max_keep_ckpts` 时，前面的权重会被删除。

```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=5, 
        max_keep_ckpts=2
    )
)
```

上述例子表示，假如一共训练 20 个 epoch，那么会在第 5, 10, 15, 20 个 epoch 保存模型，但是在第 15 个 epoch 的时候会删除第 5 个 epoch 保存的权重，在第 20 个 epoch 的时候会删除第 10 个 epoch 的权重，最终只有第 15 和第 20 个 epoch 的权重才会被保存。

〔**3. 保存最优权重**〕

如果想要保存训练过程验证集的最优权重，可以设置 `save_best` 参数，如果设置为 `'auto'`，则会根据验证集的第一个评价指标（验证集返回的评价指标是一个有序字典）判断当前权重是否最优。

```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        save_best='auto'  # 评估方式为自动（验证集的第一个评价指标）
    )
)
```

也可以直接指定 `save_best` 的值为评价指标，例如在分类任务中，可以指定为 `save_best='top-1'`，则会根据 `'top-1'` 的值判断当前权重是否最优。

除了 `save_best` 参数，和保存最优权重相关的参数还有 `rule`，`greater_keys` 和 `less_keys`，这三者用来判断 `save_best` 的值是越大越好还是越小越好。例如指定了 `save_best='top-1'`，可以指定 `rule='greater'`，则表示该值越大表示权重越好。

```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        save_best='top-1',  # 评估方式为 top-1 的值
        rule='greater'  # 越大越好
    )
)
```

〔**4. 指定保存权重的路径**〕

权重默认保存在工作目录（`work_dir`），但可以通过设置 `out_dir` 改变保存路径。

```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=5, 
        out_dir='/path/of/directory'
    )
)
```

〔**5. 制作发布用的权重**〕

如果你想在训练结束后自动生成可发布的权重（**删除不需要的权重，例如优化器状态**），你可以设置 `published_keys` 参数，选择需要保留的信息。

⚠️  **注意**：需要相应设置 `save_best` 或者 `save_last` 参数，这样才会生成可发布的权重，其中设置 `save_best` 会生成最优权重的可发布权重，设置 `save_last` 会生成最后一个权重的可发布权重，这两个参数也可同时设置。

```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1, 
        save_best='accuracy',  # best_model 的评价指标
        rule='greater',   # 评价指标越大越好
        published_keys=['meta', 'state_dict']  # 保留的key
    )
)
```

〔**6. 设置开始保存权重的 epoch 数或者 iteration 数**〕

如果想要设置控制开始保存权重的 epoch 数或者 iteration 数，可以设置 `save_begin` 参数，默认为 0，表示从训练开始就保存权重。例如，如果总共训练 10 个 epoch，并且 `save_begin` 设置为 5，则将保存第 5、6、7、8、9 和 10 个 epoch 的权重。如果 `interval=2`，则仅保存第 5、7 和 9 个 epoch 的权重。

```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=2, 
        save_begin=5
    )
)
```

### 7.3.2 LoggerHook

`LoggerHook` 负责收集日志并把日志输出到：① 终端、② 输出到文件、③ 输出到TensorBoard 等后端。

如果我们希望每迭代 20 次就输出（或保存）一次日志，我们可以设置 `interval` 参数，配置如下：

```python
default_hooks = dict(
    logger=dict(
        type='LoggerHook', 
        interval=20
    )
)
```

### 7.3.3 ParamSchedulerHook

`ParamSchedulerHook` 遍历执行器（Runner）的所有优化器参数调整策略（Parameter Scheduler）并逐个调用 `step` 方法更新优化器的参数。`ParamSchedulerHook` 默认注册到执行器并且没有可配置的参数，所以无需对其做任何配置。

### 7.3.4 IterTimerHook

`IterTimerHook` 用于记录加载数据的时间以及迭代一次耗费的时间。`IterTimerHook` 默认注册到执行器并且没有可配置的参数，所以无需对其做任何配置。

### 7.3.5 DistSamplerSeedHook

`DistSamplerSeedHook` 在分布式训练时调用 `Sampler` 的 `step` 方法以确保 `shuffle` 参数生效。`DistSamplerSeedHook` 默认注册到执行器并且没有可配置的参数，所以无需对其做任何配置。

### 7.3.6 RuntimeInfoHook

`RuntimeInfoHook` 会在执行器（Runner）的不同钩子位点将当前的运行时信息，如 `epoch`、`iter`、`max_epochs`、`max_iters`、`lr`、`metrics` 等更新至 `message hub` 中，以便其他无法访问执行器的模块能够获取到这些信息。`RuntimeInfoHook` 默认注册到执行器并且没有可配置的参数，所以无需对其做任何配置。

## 7.4 内置的自定义钩子的用法

### 7.4.1 EMAHook

`EMAHook` 在训练过程中对模型执行指数滑动平均操作，目的是提高模型的鲁棒性。

```python
custom_hooks = [
    dict(
        type='EMAHook'
    )
]

runner = Runner(custom_hooks=custom_hooks, ...)
runner.train()
```

> ⚠️  **注意**：指数滑动平均生成的模型只用于验证和测试，不影响训练。

`EMAHook` 默认使用 `ExponentialMovingAverage`，可选值还有 `StochasticWeightAverage` 和 `MomentumAnnealingEMA`。可以通过设置 `ema_type` 使用其他的平均策略。

```python
custom_hooks = [
    dict(
        type='EMAHook', 
        ema_type='StochasticWeightAverage'
    )
]
```

### 7.4.2 EmptyCacheHook

`EmptyCacheHook` 调用 `torch.cuda.empty_cache()` 释放未被使用的显存。可以通过设置 `before_epoch`, `after_iter` 以及 `after_epoch` 参数控制释显存的时机，第一个参数表示在每个 `epoch` 开始之前，第二参数表示在每次迭代之后，第三个参数表示在每个 `epoch` 之后。

```python
custom_hooks = [
    dict(
        type='EmptyCacheHook', 
        after_epoch=True  # 每一个 epoch 结束都会执行释放操作
    )
]

runner = Runner(custom_hooks=custom_hooks, ...)
runner.train()
```

### 7.4.3 SyncBuffersHook

`SyncBuffersHook` 在分布式训练每一轮（epoch）结束时同步模型的 `buffer`，例如 `BN` 层的 `running_mean` 以及 `running_var`。

```python
custom_hooks = [
    dict(
        type='SyncBuffersHook'
    )
]

runner = Runner(custom_hooks=custom_hooks, ...)
runner.train()
```

### 7.4.4 ProfilerHook

`ProfilerHook` 用于分析模型算子的执行时间以及显存占用情况。

```python
custom_hooks = [
    dict(
        type='ProfilerHook', 
        on_trace_ready=dict(type='tb_trace')
    )
]
runner = Runner(custom_hooks=custom_hooks, ...)
runner.train()
```

`profile` 的结果会保存在 `work_dirs/{timestamp}` 下的 `tf_tracing_logs` 目录，通过 `tensorboard --logdir work_dirs/{timestamp}tf_tracing_logs`进行查看。

## 7.5 自定义钩子

如果 MMEngine 提供的钩子（内置的默认钩子和内置的自定义钩子）不能满足需求，用户可以自定义钩子，只需继承钩子基类并重写相应的位点方法。

例如，如果希望在训练的过程中判断损失值是否有效，如果值为无穷大则无效，我们可以在每次迭代后判断损失值是否无穷大，因此只需重写 `after_train_iter` 位点。

```python
import torch

from mmengine.registry import HOOKS  # 钩子注册器
from mmengine.hooks import Hook  # 钩子类


@HOOKS.register_module()
class CheckInvalidLossHook(Hook):
    """Check invalid loss hook.

    This hook will regularly check whether the loss is valid
    during training.

    Args:
        interval (int): Checking interval (every k iterations).
            Defaults to 50.
    """

    def __init__(self, interval=50):
        self.interval = interval

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """All subclasses should override this method, if they need any
        operations after each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        if self.every_n_train_iters(runner, self.interval):
            assert torch.isfinite(outputs['loss']),\
                runner.logger.info('loss become infinite or NaN!')
```

我们只需将钩子的配置传给执行器（Runner）的 `custom_hooks` 的参数，执行器初始化的时候会注册钩子，便会在每次模型前向计算后检查损失值。

```python
from mmengine.runner import Runner

custom_hooks = [
    dict(
        type='CheckInvalidLossHook', 
        interval=50
    )
]
runner = Runner(custom_hooks=custom_hooks, ...)  # 实例化执行器，主要完成环境的初始化以及各种模块的构建
runner.train()  # 执行器开始训练
```

> ⚠️  自定义钩子的优先级默认为 `NORMAL (50)`，如果想改变钩子的优先级，则可以在配置中设置 `priority` 字段。也可以在定义类时给定优先级：
>
> ```python
> @HOOKS.register_module()
> class CheckInvalidLossHook(Hook):
> 
>     priority = 'ABOVE_NORMAL'
> ```