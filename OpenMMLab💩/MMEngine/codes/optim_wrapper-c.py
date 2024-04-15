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