import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR


model = torch.nn.Linear(1, 1)
dataset = [torch.randn((1, 1, 1)) for _ in range(20)]
optimizer = SGD(model, 0.1)
scheduler = ExponentialLR(optimizer, gamma=0.9)

for epoch in range(10):  # epoch
    for data in dataset:  # iteration
        optimizer.zero_grad()  # 优化器梯度清零
        output = model(data)  # 前向推理
        loss = 1 - output  # 计算损失
        loss.backward()  # 损失反向传播到优化器
        optimizer.step()  # 优化器执行参数更新
    scheduler.step()  # 调度器随着epoch的进行而执行
