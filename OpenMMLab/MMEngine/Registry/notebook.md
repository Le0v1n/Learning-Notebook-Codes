# 1. MMCV 中的 Registry

## 1.1 MMCV 简介

MMCV（Multimodal Computing and Vision Library）是一个用于计算机视觉和多模态计算的开源库。它提供了丰富的功能和工具，用于开发和研究计算机视觉任务，如图像处理、目标检测、图像分割、人脸识别等。MMCV 是由 OpenMMLab 团队开发的，它是 PyTorch 生态系统中的一个重要组成部分，可用于构建和训练深度学习模型。

MMCV 的功能包括图像和视频处理、图像和标注可视化、图像变换、多种卷积神经网络架构的实现，以及常见的 CPU 和 CUDA 操作的高质量实现。这使得它成为了计算机视觉研究和开发的有力工具。

总之，MMCV 是一个强大的库，旨在简化计算机视觉项目的开发和研究工作，提供了许多便捷的工具和功能。它通常与 PyTorch 结合使用，为视觉任务提供了丰富的支持。

具体来说，它提供了以下功能：

- [图像和视频处理](https://mmcv.readthedocs.io/zh_CN/latest/understand_mmcv/data_process.html)
- [图像和标注结果可视化](https://mmcv.readthedocs.io/zh_CN/latest/understand_mmcv/visualization.html)
- [图像变换](https://mmcv.readthedocs.io/zh_CN/latest/understand_mmcv/data_transform.html)
- [多种 CNN 网络结构](https://mmcv.readthedocs.io/zh_CN/latest/understand_mmcv/cnn.html)
- [高质量实现的常见 CUDA 算子](https://mmcv.readthedocs.io/zh_CN/latest/understand_mmcv/ops.html)

MMCV 支持多种平台，包括：

- Linux
- Windows
- macOS

如想了解更多特性和使用，请参考[文档](http://mmcv.readthedocs.io/zh_CN/latest)。

> **注意**: MMCV 需要 Python 3.7 以上版本。

## 1.2 MMCV 安装

**官网**: [https://github.com/open-mmlab/mmcv](https://github.com/open-mmlab/mmcv)

有两个版本的 MMCV：

1. mmcv：全面的版本，内置了丰富的功能和各种 CUDA 操作。构建时间较长。
2. mmcv-lite：精简版本，没有 CUDA 操作，但具备所有其他功能，类似于 mmcv<1.0.0 版本。在不需要 CUDA 操作时，这个版本非常有用。

注意：不要在同一个环境中同时安装这两个版本，否则可能会遇到类似 ModuleNotFound 的错误。在安装另一个版本之前，你需要卸载一个版本。如果可用的话，强烈建议安装全版本，特别是如果你有 CUDA 支持的话。

安装命令如下：

```bash
# 完整版安装
pip install -U openmim
mim install mmcv

# 精简版安装(没有 CUDA 且没有 Registry)
pip install -U openmim
mim install mmcv-lite
```

**注意**：mmcv 2.0 以上版本我没有找到 `Registry`（恕我笨比了），所以这里建议大家使用 1.x 版本。

```bash
# 带有 CUDA
pip install -U openmim
mim install mmcv-full==1.7.0

# 不带 CUDA
pip install -U openmim
mim install mmcv==1.7.0
```

> 1. 在 2.x 版本中，mmcv 即为带有 CUDA 版本，而在 1.x 版本中，mmcv 对应 2.x 版本中的 mmcv-lite，而 mmcv-full 才对应 2.x 版本中的 mmcv
> 2. 速度慢可以换源: `-i https://pypi.tuna.tsinghua.edu.cn/simple` (pip 和 min 都支持)

## 1.3 MMCV 中的 Registry

注册机制是 MMCV 中非常重要的一个概念，在 MMDetection 中，如果想要增加自己的算法模块或流程，都需要通过注册机制来实现。

介绍注册机制之前先介绍一下 Registry 类。

MMCV 使用注册器(`Registry`)来管理具有相似功能的不同模块，比如 [ResNet](https://blog.csdn.net/weixin_44878336/article/details/124517277)、[FPN](https://blog.csdn.net/weixin_44878336/article/details/126004264)、RoIHead 都属于模型结构，[SGD](https://blog.csdn.net/weixin_44878336/article/details/124869795)、[Adam](https://blog.csdn.net/weixin_44878336/article/details/124869795) 都属于[优化器](https://blog.csdn.net/weixin_44878336/article/details/124869795)。

注册器内部其实是在维护一个全局的查询表，`key` 是字符串，`value` 是类。简单来说，注册器可以看做字符串到类(Class)的映射 (`{str: class}`)。借助注册器，用户可以通过字符串查询到对应的类，并实例化该类。有了这个认知后，再看 Registry 类的源码就很容易理解。

先看下构造函数，其功能主要是：
1. 初始化注册器的名字
2. 实例化函数

现在我们初始化一张字典类型的查询表 `_module_dict`，代码如下：

```python
from mmcv.utils import Registry

class Registry:
    # 构造函数
    def __init__(self, name, build_func=None, parent=None, scope=None):
        """
        name (str): 注册器的名字
        build_func(func): 从注册器构建实例的函数句柄
        parent (Registry): 父类注册器
        scope (str): 注册器的域名
        """
        self._name = name
        
        # 使用 module_dict 管理字符串到类的映射 {'str': class}
        self._module_dict = dict()
        self._children = dict()
```

### 1.3.1 举例：调用 Registry 注册实例 MODELS

比如说，我们现在想要使用注册器来管理我们的模型，首先初始化一个 `Registry` 实例 `MODELS`，然后调用 `Registry` 类的 `register_module()` 方法完成 `ResNet` 和 `VGG` 类的注册，可以看到最后 `MODELS` 的打印结果中包含了这两个类的信息(打印信息中 `items` 对应的其实就是 `self._module_dict`)，表示注册成功。

为了代码简洁，还是使用 `@` 实现 `register_module()` 的调用。然后就可以通过 `build()` 函数来实例化我们的模型了。

```python

```



# 知识来源
1. 