# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
YOLO-specific modules.

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import (
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DetectMultiBackend,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto,
)
from models.experimental import MixConv2d
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, colorstr, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    profile,
    scale_img,
    select_device,
    time_sync,
)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # 在整个构建过程中计算出来的 strides 大小
    dynamic = False  # 是否强制重构 grid
    export = False  # 是否是导出模式

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        """Detect 检测头这个类的初始化方法

        Args:
            nc (int, optional): 数据集类别数. Defaults to 80. 例子：80
            anchors (tuple, optional): 先验框的尺寸. Defaults to (). 例子：[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
            ch (tuple, optional): 预测特征图的通道数. Defaults to (). 例子：[128, 256, 512]
            inplace (bool, optional): 是否使用原地操作. Defaults to True. 例子：True
        """
        super().__init__()
        self.nc = nc  # 数据集的类别数
        self.no = nc + 5  # 每个 Anchor 输出的数量，例子：COCO数据集则是 80 + 5
        self.nl = len(anchors)  # 预测特征图的数量，例子：3
        self.na = len(anchors[0]) // 2  # 每个Anchor的数量，具体来说就是每个Anchor的尺寸的数量，默认每个Anchor有3种尺寸，例：(10, 13), (16, 30), (33, 23)，这是三种尺寸
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # 初始化网格（grid）。torch.empty(0)会创建一个没有任何元素且未初始化的Tensor，其值为tensor([])，shape为torch.Size([0])
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # 初始化Anchor的网格
        # 将先验框的信息（Anchors）持久化到模型中（注册为一个名为anchors的缓冲区）。这样，anchors就会成为模块状态的一部分，会在模型保存和加载时一起保存和加载，但是它不会被视为模型的参数，因此不会在模型训练过程中被更新。
        """💡  关于 self.register_buffer() 的说明：
            模型中需要保存下来的参数包括两种：
                ①反向传播需要被optimizer更新的，称之为parameter。
                ②反向传播不需要被optimizer更新的，称之为buffer
            对于②，我们在创建Tensor之后需要使用register_buffer()这个方法将其注册为buffer，不然默认是parameter。
            注册的buffer我们可以通过，model.buffers()返回，注册完后参数也会自动保存到OrderDict中区。
            注意：buffer的更新在forward中，optimizer.step()只更新nn.parameter类型的参加，不会更新buffer
        """
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl, na, 2): [预测特征图数量，Anchor数量，坐标(xy)]
        self.m = nn.ModuleList(
            nn.Conv2d(
                in_channels=x, 
                out_channels=self.no * self.na,   # 85*3=255
                kernel_size=1
            ) for x in ch  # 预测特征图通道数[128, 256, 512]
        )  # output conv，1x1卷积
        """
            ModuleList(
                (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
                (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
                (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
            )
        """
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        """x：三个预测特征图（值均为0）。
            len(x) = 3
            x[0].shape = torch.Size([1, 128, 32, 32])
            x[1].shape = torch.Size([1, 256, 16, 16])
            x[2].shape = torch.Size([1, 512,  8,  8])
        """
        z = []  # 存放前向推理的结果
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # 经过一个卷积：[1, 128, 32, 32] -> [1, 255, 32, 32]
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # [1, 255, 32, 32] -> [1, 3, 32, 32, 85]，其中255=每个Anchor输出的数量（80+5=85），3=每个Anchor的尺寸的数量
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # 如果不是训练状态
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        # return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
        if self.training:
            # 如果模块处于训练模式，直接返回x
            """len(x) = 3
                x[0].shape = torch.Size([1, 3, 32, 32, 85])
                x[1].shape = torch.Size([1, 3, 16, 16, 85])
                x[2].shape = torch.Size([1, 3, 8, 8, 85])
            """
            return x
        else:
            # 如果模块不处于训练模式，进一步检查self.export的值
            if self.export:
                # 如果处于导出模式，只返回拼接后的z
                return torch.cat(z, 1)
            else:
                # 如果既不是训练模式也不是导出模式，返回拼接后的z和x
                return torch.cat(z, 1), x

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """生成一个网格坐标和一个Anchor网格

        Args:
            nx (int, optional): 网格的x方向上的点数. Defaults to 20.
            ny (int, optional): 网格的y方向上的点数. Defaults to 20.
            i (int, optional): 用于选择特定的Anchor. Defaults to 0.
            torch_1_10 (_type_, optional): 用于检查PyTorch的版本是否大于或等于1.10.0. Defaults to check_version(torch.__version__, "1.10.0").

        Returns:
            _type_: _description_
        """
        # 获取Anchor Tensor的设备和数据类型。这些信息将用于创建新的 Tensor，以确保它们在与Anchor Tensor相同的设备和数据类型上
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        
        # 定义网格的形状，其中na表示每个Anchor的尺寸的数量，默认为3，表示有大中小3种大小
        shape = 1, self.na, ny, nx, 2  # grid shape
        
        # 创建了两个一维 Tensor，分别包含ny和nx个元素，这些元素是从0到ny-1和nx-1的整数。
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        
        # 创建一个二维网格坐标。torch.meshgrid函数从给定的一维坐标 Tensor生成二维网格坐标。如果PyTorch版本大于或等于1.10.0，使用indexing="ij"来确保索引的顺序与NumPy兼容
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        
        # x和y坐标堆叠成一个五维 Tensor，并将其形状扩展为之前定义的shape（1, self.na, ny, nx, 2）。
        # 然后，它从每个坐标中减去0.5，这是因为在目标检测中，我们通常希望Anchor位于像素的中心而不是左上角。
        """tensor.expand()方法说明：
            在PyTorch中，tensor.expand方法用于扩展一个 Tensor（tensor），它会返回一个新的 Tensor，该 Tensor的特定维度被扩展了。
            这个方法允许我们在不复制底层数据的情况下，创建一个在指定维度上具有更大尺寸的新 Tensor。
            expand方法接受一个或多个参数，这些参数指定了 Tensor在每个维度上的扩展尺寸。
            与torch.Tensor.view不同，expand不会改变 Tensor中元素的数量，也不会复制数据。
            相反，它创建了一个新的“视图”（view），这个视图在内存中与原始 Tensor共享相同的数据。
            这意味着对扩展后的 Tensor进行的任何修改都会反映到底层数据上，反之亦然。

            expand方法的一个常见用途是在需要进行广播操作时，将 Tensor的尺寸扩展到与其他 Tensor兼容。
            例如，如果我们有一个批量大小为1的 Tensor，并希望将其与批量大小为N的 Tensor进行运算，
            我们可以使用expand方法将第一个 Tensor的批量大小扩展到N，这样就可以进行元素级别的运算了。
            
            例子：
                >>> x = torch.tensor([[1, 2, 3]])
                
                >>> print(x)
                tensor([[1, 2, 3]])
                
                >>> print(x.size())
                torch.Size([1, 3])
                
                >>> y = x.expand(2, 3)
                
                >>> print(y.size())
                torch.Size([2, 3])
                
                >>> print(y)
                tensor([[1, 2, 3],
                        [1, 2, 3]])
                        
                >>> y = x.expand(3, 3)
                >>> print(y)
                tensor([[1, 2, 3],
                        [1, 2, 3],
                        [1, 2, 3]])
        """
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        
        # 创建了一个Anchor网格。self.anchors[i]是特定索引i的Anchor坐标，self.stride[i]是与这些Anchor相关联的步长。
        # 这个步长用于根据特征图的分辨率调整Anchor的大小。然后，这个Anchor Tensor被重新塑形并扩展到与网格相同的形状。
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    # YOLOv5 base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs, delta_time
        for m in self.model:  # 遍历模型的所有模块
            if m.f != -1:  # 如果该模块并不是来自之前的层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:  # 是否进行该模块的性能评估（models/yolo.py可用）
                self._profile_one_layer(m, x, dt)
            x = m(x)  # 使用该模块进行推理
            y.append(x if m.i in self.save else None)  # 如果模块的索引在 self.save 中，则保存该特征图，例子：self.save=[4, 6, 10, 14, 17, 20, 23]
            if visualize:  # 是否进行可视化（detect.py可用）
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        """用于分析模型中某一层的计算复杂度和运行时间

        Args:
            m (_type_): 模型的某一层
            x (_type_): 模型的某一层的输入特征图
            dt (_type_): delta time, 时间差
        """
        # 判断被测试的模块是否为模型的最后一层，如果是则复制输入张量以防止在原地操作时修改原始输入
        c = m == self.model[-1]
        
        # 使用thop工具计算层的浮点运算次数（FLOPs），单位为十亿（GFLOPs）（💡  如果thop不可用，则设置为0）
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()  # 等待所有GPU都计算完毕

        # 重复运行层10次以计算平均运行时间
        for _ in range(10):
            m(x.copy() if c else x)  # 如果是最后一层，则使用复制的输入，否则使用原始输入
        dt.append((time_sync() - t) * 100)  # 计算运行时间并添加到dt列表中，单位为毫秒

        # 如果是模型的第一层，打印表头
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        # 打印层的运行时间、浮点运算次数和参数数量
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")

        # 如果是最后一层，打印总运行时间和总参数数量
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")


    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        """DetectionModel的初始化方法

        Args:
            cfg (str, optional): 模型的配置文件路径. Defaults to "yolov5s.yaml".
            ch (int, optional): 模型的输入特征图通道数（输入图片通道数）. Defaults to 3.
            nc (_type_, optional): 数据集类别数. Defaults to None.
            anchors (_type_, optional): 先验框. Defaults to None.
        """
        super().__init__()
        if isinstance(cfg, dict):  # 判断 cfg 是否为一个字典
            self.yaml = cfg  # model dict
        else:  # 否则认为它是一个 .yaml 文件
            import yaml  # for torch hub

            # Path(cfg)创建了一个Path对象，其路径由变量cfg指定。然后，它调用这个Path对象的name属性，该属性返回路径的最后一部分，即文件名。
            # 举个例子，如果cfg变量的值是"/path/to/config.yaml"，那么self.yaml_file将会被赋值为"config.yaml"，即去除了路径部分的文件名。
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # 此时 self.yaml 变成了一个字典，它的keys()=dict_keys(['nc', 'depth_multiple', 'width_multiple', 'anchors', 'backbone', 'head'])
                """self.yaml
                {
                    'nc': 80, 
                    'depth_multiple': 0.33, 
                    'width_multiple': 0.5, 
                    'anchors': [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
                    'backbone': [[-1, 1, 'Conv', [...]], [-1, 1, 'Conv', [...]], [-1, 3, 'C3', [...]], [-1, 1, 'Conv', [...]], [-1, 6, 'C3', [...]], [-1, 1, 'Conv', [...]], [-1, 9, 'C3', [...]], [-1, 1, 'Conv', [...]], [-1, 3, 'C3', [...]], [-1, 1, 'SPPF', [...]]]
                    'head': [[-1, 1, 'Conv', [...]], [-1, 1, 'nn.Upsample', [...]], [[...], 1, 'Concat', [...]], [-1, 3, 'C3', [...]], [-1, 1, 'Conv', [...]], [-1, 1, 'nn.Upsample', [...]], [[...], 1, 'Concat', [...]], [-1, 3, 'C3', [...]], [-1, 1, 'Conv', [...]], [[...], 1, 'Concat', [...]], [-1, 3, 'C3', [...]], [-1, 1, 'Conv', [...]], [[...], 1, 'Concat', [...]], [-1, 3, 'C3', [...]], ...]
                }
                """

        # 定义模型
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # 输入通道数（一般都是3 -> RGB），这里用的是dict.get(key, default)方法
        if nc and nc != self.yaml["nc"]:  # 如果使用这个类时传入了nc，且与配置文件中的nc有冲突：使用nc而非self.yaml["nc"]，并将self.yaml["nc"]重新赋值为nc
            LOGGER.info(f"使用 {nc = } 覆盖 model.yaml 中的 {self.yaml['nc'] = }")
            self.yaml["nc"] = nc  # override yaml value
        # 💡  如果anchors为None，也不会进行赋值！
        if anchors:  # 如果使用这个类时传入了anchors，则使用传入的anchors而非self.yaml["anchors"]，并使用anchors覆盖self.yaml["anchors"]
            LOGGER.info(f"使用 {anchors = } 覆盖 model.yaml 中的 anchors")
            self.yaml["anchors"] = round(anchors)  # override yaml value
            
        # 通过 parse_model() 函数来解析 model.yaml 文件并构建模型以及推理时需要保存特征图的Module索引（self.save）
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        # 构建 strides, anchors
        m = self.model[-1]  # 获取 Detect() 部分
        """m 模块，即 Detect 的结构如下：
            Detect(
            (m): ModuleList(
                (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
                (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
                (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
            )
            )
        """
        if isinstance(m, (Detect, Segment)):  # 判断是否取出的 self.model[-1] 是 Detect 或者 Segment 模块
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)  # 这是一个 lambda 函数
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward: tensor([ 8., 16., 32.])

            # 检查anchor顺序和stride顺序是否一致
            check_anchor_order(m)  

            # 计算anchor大小，例子：[10, 13] -> [1.25, 1.625]
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # 初始化偏置，only run once

        # Init weights, biases
        initialize_weights(self)  # 初始化权重
        self.info()
        LOGGER.info("")


    def forward(self, x, augment=False, profile=False, visualize=False):
        # 💡  注意：这里的 augment 不是 Data Augmentation，而是有没有开启 TTA（Test Time Augmentation）
        #           具体参数为 --augment，此时 --imgsz 832（💡  开启TTA后图片尺寸也应该增大）
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train


    def _forward_augment(self, x):
        """使用 TTA 的推理

        Args:
            x (Tensor): 输入图片，shape为[B, 3, imgsz, imgsz]

        Returns:
            _type_: 使用TTA的模型推理结果
        """
        img_size = x.shape[-2:]  # height, width，例子：torch.Size([576, 864])
        s = [1, 0.83, 0.67]  # scales，TTA默认使用的三个图片的尺寸
        f = [None, 3, None]  # flips (2-ud, 3-lr)，其中2表示上下的flip，3为左右的flip，None表示不进行flip
        y = []  # outputs，接收TTA推理结果
        for si, fi in zip(s, f):  # si: scale_i, fi: flip_i
            xi = scale_img(
                # tensor.flip(dim)：沿着指定的维度将张量中的元素顺序颠倒。对于我们的图片（B, C, H, W）而言，img.flip(2)是高度翻转，即上下翻转；img.flip(3)是水平翻转。
                img=x.flip(fi) if fi else x,
                ratio=si, 
                gs=int(self.stride.max())  # gs: grid size
            )
            
            # 使用模型对新的xi进行推理
            yi = self._forward_once(xi)[0]  # forward    例子：torch.Size([16, 21735, 85])
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            
            # 对结果进行反scale处理
            yi = self._descale_pred(yi, fi, si, img_size)  # 例子：torch.Size([16, 21735, 85])
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train


    def _descale_pred(self, p, flips, scale, img_size):
        """在增强推理后对预测进行逆缩放（逆操作）

        Args:
            p (_type_): 预测结果张量，包含边界框的坐标和宽度、高度    例子：torch.Size([16, 21735, 85])
            flips (_type_): 指示图像是否进行了水平或垂直翻转的整数，2表示垂直翻转，3表示水平翻转    例子：2
            scale (_type_): 图像放缩的比例因子    例子：0.83
            img_size (_type_): 原始图像的大小，形式为(高度, 宽度)    例子：torch.Size([576, 864])

        Returns:
            _type_: 返回经过逆操作的预测结果张量
        """
        if self.inplace:  # 如果inplace为True，直接在原张量上进行操作以节省内存
            p[..., :4] /= scale  # 将边界框的坐标和宽度、高度除以放缩比例，以逆放缩操作
            if flips == 2:  # 如果图像进行了垂直翻转，则对y坐标进行逆翻转操作
                p[..., 1] = img_size[0] - p[..., 1]  # 使用图像的高度减去y坐标
            elif flips == 3:  # 如果图像进行了水平翻转，则对x坐标进行逆翻转操作
                p[..., 0] = img_size[1] - p[..., 0]  # 使用图像的宽度减去x坐标
        else:  # 如果inplace为False，创建新的张量以存储逆操作的结果
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # 逆放缩操作
            if flips == 2:  # 如果图像进行了垂直翻转，则对y坐标进行逆翻转操作
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:  # 如果图像进行了水平翻转，则对x坐标进行逆翻转操作
                x = img_size[1] - x  # de-flip lr
            # 将逆放缩和逆翻转后的结果拼接回预测张量中
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        """在YOLOv5模型的增强推理过程中裁剪掉多余的预测尾部

        Args:
            y (list): 增强推理的输出，一个包含多个检测层预测的张量列表

        Returns:
            list: 裁剪掉多余尾部的输出list
        """
        # 获取检测层的数量，通常对应于不同的特征图层级（如P3, P4, P5）    例子：3
        nl = self.model[-1].nl
        
        # 计算每个检测层网格点的总数，每个层级的网格点数是4的x次幂，总和即为所有层级的网格点数
        g = sum(4**x for x in range(nl))  # grid points

        # 设置一个排除层计数器，用于后续计算要排除的预测尾部的数量
        e = 1  # exclude layer count
        
        # 计算要排除的预测尾部的索引，这里计算的是第一个检测层（最大特征图层级）的索引
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        
        # 从第一个检测层的预测中排除掉尾部，保留较大的预测目标
        y[0] = y[0][:, :-i]  # large
        
        # 计算要排除的预测尾部的索引，这里计算的是最后一个检测层（最小特征图层级）的索引
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        
        # 从最后一个检测层的预测中排除掉头部，保留较小的预测目标
        y[-1] = y[-1][:, i:]  # small
        
        # 返回裁剪后的预测张量列表
        return y

    def _initialize_biases(self, cf=None):  # ，其中cf is 
        """初始化Detect()模块的偏置
            self: 实例化对象
            cf: class frequency，类别频率，它表示每个类别的数量
        
        此函数出处为：[RetinaNet](https://arxiv.org/abs/1708.02002) section 3.3
        cf计算方式：`cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.`
        """
        # 遍历Detect()模块中的每个检测层（m.m）及其步长（m.stride）
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            # 将卷积层的偏置（bias）从(255)转换为(3,85)的形状
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)

            # 初始化目标偏置（obj），8个对象在640x640的图像中
            # （这里的8是根据RetinaNet的论文中提到的，每个尺度上平均有8个对象）
            # （这里的0.6是根据RetinaNet的论文中提到的，在所有类别中，大约有60%的类别的对象数量是最大的）
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            
            # 初始化分类偏置（cls），如果cf为None，则使用均匀分布的类频率；如果cf不为None，则使用实际的类频率
            b.data[:, 5 : 5 + m.nc] += (math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum()))  # cls

            # 将调整后的偏置设置回卷积层（requires_grad=True表示这个参数可以计算梯度，即在训练过程中可以更新这个参数）
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg="yolov5s-seg.yaml", ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, "models.common.Classify"  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []  # 推理时需要保存特征图的Module索引（self.save）
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None


def parse_model(d, ch):  # model_dict, input_channels(3)
    """通过解析 model.yaml 文件从而构建模型

    Args:
        d (dict): 模型字典
        ch (int): 输入图像的通道数，一般为3

    Returns:
        _type_: _description_
    """
    # Parse a YOLOv5 model.yaml dictionary
    #                  from  n    params  module                                  arguments
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act, ch_mul = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],        # 模型深度: gd = global depth
        d["width_multiple"],        # 模型宽度: gw = global width
        d.get("activation"),        # 获取激活函数，没有则为 None
        d.get("channel_multiple"),  # 获取 channel_multiple 系数，没有则为 None
    )
    
    if act:  # 如果 model.yaml 文件中定义了 "activation"
        Conv.default_act = eval(act)  # 重新定义默认的激活函数, 例如 Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('根据 model.yaml 文件，重新定义默认的激活函数为:')} {act}")  # print
    if not ch_mul:  # 如果 model.yaml 文件中没有定义 "channel_multiple"
        ch_mul = 8  # 让 channel_multiple 默认为 8

    # na: anchors尺寸的种类，一般为3
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors

    no = na * (nc + 5)  # 每个预测特征图的输出通道数，number of outputs = anchors * (classes + 5)，例子：255 = 3 * (80 + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # 对 backbone 和 head 中的所有层进行遍历
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  
        # f <-> from：表示输入的来源。-1 表示前一层的输出作为输入，例子：-1
        # n <-> number：表示重复使用该模块的次数，例子：1
        # m <-> module：表示使用的特征提取模块类型，例子：Conv
        # args：表示模块的参数，例子：[64, 6, 2, 2]

        # 将字符串转换为对应的代码名称（不懂的看一下 eval 函数），例子：'Conv' -> <class 'models.common.Conv'>
        m = eval(m) if isinstance(m, str) else m  

        # 遍历每一层的参数args，目的是防止参数中出现字符串（将字符串都转换为int）
        for j, a in enumerate(args):  # j: 参数的索引   a: 具体的参数
            # with contextlib.suppress(...): 是Python中的一个上下文管理器，用于抑制在代码块执行过程中发生的特定异常。
            # 在这里，它用于抑制NameError异常。
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        # 先将所有的 number 乘上 深度系数
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain

        # 判断当前模块是否在这个字典中
        if m in {
            Conv,                # 普通的卷积层
            GhostConv,           # 华为在 GhostNet 中提出的 Ghost 卷积
            Bottleneck,          # ResNet 同款
            GhostBottleneck,     # 将其中的 3x3 卷积替换为 GhostConv
            SPP,                 # Spatial Pyramid Pooling
            SPPF,                # SPP + Conv
            DWConv,              # 深度卷积
            MixConv2d,           # 一种多尺度卷积层，可以在不同尺度上进行卷积操作。它使用多个不同大小的卷积核对输入特征图进行卷积，并将结果进行融合
            Focus,               # 一种特征聚焦层，用于减少计算量并增加感受野。它通过将输入特征图进行通道重排和降采样操作，以获取更稠密和更大感受野的特征图
            CrossConv,           # 一种交叉卷积层，用于增加特征图的多样性。它使用不同大小的卷积核对输入特征图进行卷积，并将结果进行融合
            BottleneckCSP,       # 一种基于残差结构的卷积块，由连续的 Bottleneck 模块和 CSP（Cross Stage Partial）结构组成，用于构建深层网络，提高特征提取能力
            C3,                  # 一种卷积块，由三个连续的卷积层组成。它用于提取特征，并增加网络的非线性能力
            C3TR,                # C3TR 是 C3 的变体，它在 C3 的基础上添加了 Transpose 卷积操作。Transpose 卷积用于将特征图的尺寸进行上采样
            C3SPP,               # C3SPP 是 C3 的变体，它在 C3 的基础上添加了 SPP 操作。这样可以在不同尺度上对特征图进行池化，并增加网络的感受野
            C3Ghost,             # C3Ghost 是一种基于 C3 模块的变体，它使用 GhostConv 代替传统的卷积操作
            nn.ConvTranspose2d,  # 转置卷积
            DWConvTranspose2d,   # DWConvTranspose2d 是深度可分离的转置卷积层，用于进行上采样操作。它使用逐点卷积进行特征图的通道之间的信息整合，以减少计算量
            C3x,                 # C3x 是一种改进的 C3 模块，它在 C3 的基础上添加了额外的操作，如注意力机制或其他模块。这样可以进一步提高网络的性能
        }:
            c1, c2 = ch[f], args[0]  # c1: 卷积的输入通道数, c2: 卷积的输出通道数 | ch[f] 上一次的输出通道数（即本层的输入通道数），args[0]：配置文件中想要的输出通道数
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, ch_mul)  # 让输出通道数*width_multiple

            args = [c1, c2, *args[1:]]  # 此时的c2已经是修改后的c2乘上width_multiple的c2了 | *args[1:]将其他非输出通道数的参数解包

            # 如果当前层是 BottleneckCSP, C3, C3TR, C3Ghost, C3x 中的一种（这些结构都有 Bottleneck 结构）
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats | 需要让Bottleneck重复n次
                n = 1  # 重置n（其他层没有 Bottleneck 的模块不需要重复）
        
        elif m is nn.BatchNorm2d:  # 如果是BN层
            args = [ch[f]]  # 确定输出通道数
        
        elif m is Concat:  # 如果是 Concat 层
            c2 = sum(ch[x] for x in f)  # Concat是按着通道维度进行的，所以通道会增加
        
        elif m in {Detect, Segment}:  # 如果模块是 Detect 模块或者是 Segment 模块
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, ch_mul)
                
        elif m is Contract:  # 如果是 Contract 模块
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:  # 如果是 Expand 模块
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        # 将所有的模块都解包出来，用nn.Sequential接收
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module

        # 将模块名字中__main__.字符串删除
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        
        # 统计模块中的参数数量
        np = sum(x.numel() for x in m_.parameters())  # number params
        
        # 修改nn.Sequential格式的模块的属性
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # i: index    f: from    t: type    np: number of parameters
        #   0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]
        LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        
        if i == 0:  # 如果是第一层
            ch = []
        ch.append(c2)
        
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_true", help="profile model speed")
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")
    parser.add_argument("--test", action="store_true", help="test all yolo*.yaml")
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / "models").rglob("yolo*.yaml"):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f"Error in {cfg}: {e}")

    else:  # report fused model summary
        model.fuse()
