import torch
import torch.nn as nn


maxpools = [nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in range(3, 15, 2)]

input_data = torch.randn(1, 1, 4, 4)
print("输入特征图大小: ", input_data.size())

for maxpool in maxpools:
    output_data = maxpool(input_data)
    print("输出特征图大小: ", output_data.size())
