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
