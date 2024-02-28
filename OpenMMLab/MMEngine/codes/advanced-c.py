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
