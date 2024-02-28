# 1. 注册器（REGISTRY）

OpenMMLab 的算法库支持了丰富的算法和数据集，因此实现了很多功能相近的模块。例如 ResNet 和 SE-ResNet 的算法实现分别基于 ResNet 和 SEResNet 类，这些类有相似的功能和接口，都属于算法库中的模型组件。为了管理这些功能相似的模块，MMEngine 实现了 注册器。OpenMMLab 大多数算法库均使用注册器来管理它们的代码模块，包括 MMDetection， MMDetection3D，MMPretrain 和 MMagic 等。

# 2. 什么是注册器

MMEngine 实现的注册器可以看作一个映射表和模块构建方法（build function）的组合。

映射表维护了一个<font color='red'><b>字符串（`str`）到类（`class`）或者函数（`function`）的映射</b></font>，使得用户可以借助字符串（`str`）查找到相应的类（`class`）或函数（`function`）。

例如维护字符串 "ResNet" 到 ResNet 类或函数的映射，使得用户可以通过 "ResNet" 字符串找到 ResNet 类。

---

模块构建方法（build function）则定义了如何根据字符串（`str`）查找到对应的类（`class`）或函数（`function`）以及如何实例化这个类（`class`）或者调用这个函数（`function`）。

例如：

- 通过字符串 `"bn"` 找到 `nn.BatchNorm2d` 并实例化 `BatchNorm2d` 模块；
- 通过字符串 `"build_batchnorm2d"` 找到 `build_batchnorm2d` 函数并返回该函数的调用结果。

> 💡 MMEngine 中的注册器默认使用 `build_from_cfg` 函数来查找并实例化字符串（`str`）对应的类（`class`）或者函数（`function`）。

<div align=center>
    <img src=./imgs_markdown/Registry的映射表和构建方法.jpg
    width=65%>
    <center></center>
</div>

一个注册器管理的类（`class`）或函数（`function`）通常有相似的接口和功能，因此该注册器可以被视作这些类（`class`）或函数（`function`）的抽象。例如注册器 `MODELS` 可以被视作所有模型的抽象，管理了 ResNet，SEResNet 和 RegNetX 等分类网络的类（`class`）以及 `build_ResNet`, `build_SEResNet` 和 `build_RegNetX` 等分类网络的构建函数。

# 3. 入门用法

使用注册器管理代码库中的模块，需要以下三个步骤。

1. 创建注册器
2. 创建一个用于实例化类的构建方法（可选，在大多数情况下可以只使用默认方法）
3. 将模块加入注册器中

## 3.1 激活函数示例

假设我们要实现一系列激活模块并且希望仅修改配置就能够使用不同的激活模块而无需修改代码。

### 3.1.1 Step1：创建注册器

```python
from mmengine import Registry

ACTIVATION = Registry(
    'activation', 
    scope='mmengine',  # scope 表示注册器的作用域，如果不设置，默认为包名，例如在 mmdetection 中，它的 scope 为 mmdet
    locations=['mmengine.models.activation']  # locations 表示注册在此注册器的模块所存放的位置，注册器会根据预先定义的位置在构建模块时自动 import
)
```

`locations` 指定的模块 `mmengine.models.activations` 对应了 `mmengine/models/activations.py` 文件。在使用注册器构建模块的时候，`ACTIVATION` 注册器会自动从该文件中导入实现的模块。

### 3.1.2 Step2：实现激活函数

因此，我们可以在 `mmengine/models/activations.py` 文件中实现不同的激活函数，例如 Sigmoid，ReLU 和 Softmax。

```python
from mmengine import Registry
import torch.nn as nn


ACTIVATION = Registry(
    'activation', 
    scope='mmengine',  # scope 表示注册器的作用域，如果不设置，默认为包名，例如在 mmdetection 中，它的 scope 为 mmdet
    locations=['mmengine.models.activation']  # locations 表示注册在此注册器的模块所存放的位置，注册器会根据预先定义的位置在构建模块时自动 import
)


# 使用注册器管理模块
@ACTIVATION.register_module()
class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        print("Call Sigmoid.forward")
        return x
    
@ACTIVATION.register_module()
class ReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, x):
        print('call ReLU.forward')
        return x

@ACTIVATION.register_module()
class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print('call Softmax.forward')
        return x
    
    
for k, v in ACTIVATION.module_dict.items():
    print(f"{k}: {v}")
```

```
Sigmoid: <class '__main__.Sigmoid'>
ReLU: <class '__main__.ReLU'>
Softmax: <class '__main__.Softmax'>
```

使用注册器管理模块的关键步骤是，将实现的模块注册到注册表 `ACTIVATION` 中。通过 `@ACTIVATION.register_module()` 装饰所实现的模块，字符串和类或函数之间的映射就可以由 `ACTIVATION` 构建和维护，我们也可以通过 `ACTIVATION.register_module(module=ReLU)` 实现同样的功能。

通过注册，我们就可以通过 ACTIVATION 建立字符串与类或函数之间的映射。

---

⚠️ 注意：只有模块所在的文件被导入时，注册机制才会被触发，用户可以通过三种方式将模块添加到注册器中：

1. 在 `locations` 指向的文件中实现模块。注册器将自动在预先定义的位置导入模块。这种方式是为了简化算法库的使用，以便用户可以直接使用 `REGISTRY.build(cfg)`。

2. 手动导入文件。常用于用户在算法库之内或之外实现新的模块。

3. 在配置中使用 `custom_imports` 字段。 详情请参考导入自定义 Python 模块。

### 3.1.3 Step3：使用注册的模块

模块成功注册后，我们可以通过配置文件使用这个激活模块。

https://mmengine.readthedocs.io/zh-cn/latest/tutorials/runner.html










