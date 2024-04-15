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
