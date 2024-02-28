import torch.nn as nn
from torch.optim import SGD
from mmengine.optim import OptimWrapper, AmpOptimWrapper


model = nn.Linear(1, 1)
optimizer = SGD(model.parameters(), lr=0.01)

optim_wrapper = OptimWrapper(optimizer=optimizer)  # MMEngine普通的优化器
# amp_optim_wrapper = AmpOptimWrapper(optimizer=optimizer)  # MMEngine使用AMP的优化器

# 导出状态字典
optim_state_dict = optim_wrapper.state_dict()
# amp_optim_state_dict = amp_optim_wrapper.state_dict()

print(f"{optim_state_dict = }")
# print(f"{amp_optim_state_dict = }")

optim_wrapper_new = OptimWrapper(optimizer=optimizer)
# amp_optim_wrapper_new = AmpOptimWrapper(optimizer=optimizer)

# 加载状态字典
# amp_optim_wrapper_new.load_state_dict(amp_optim_state_dict)
optim_wrapper_new.load_state_dict(optim_state_dict)