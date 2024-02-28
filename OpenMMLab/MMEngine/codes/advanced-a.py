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
