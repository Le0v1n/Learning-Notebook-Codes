import torch.nn as nn
from torch.optim import SGD

from mmengine.optim import OptimWrapper

model = nn.Linear(1, 1)
optimizer = SGD(model.parameters(), lr=0.01)  # PyTorch的优化器定义方法
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