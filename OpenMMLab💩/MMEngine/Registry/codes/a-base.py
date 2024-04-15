import os
import sys
import torch
import torch.nn as nn
from mmengine import Registry

sys.path.append(os.getcwd())
from utils.outer import xprint


ACTIVATION = Registry(
    'activation', 
    scope='mmengine',  # scope 表示注册器的作用域，如果不设置，默认为包名，例如在 mmdetection 中，它的 scope 为 mmdet
    locations=['mmengine.models.activations']  # locations 表示注册在此注册器的模块所存放的位置，注册器会根据预先定义的位置在构建模块时自动 import
)


# 使用注册器管理模块
@ACTIVATION.register_module()
class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        print("Call Sigmoid.forward")
        return x
    
@ACTIVATION.register_module()
class ReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, x):
        print('call ReLU.forward')
        return x

@ACTIVATION.register_module()
class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print('call Softmax.forward')
        return x
    
    
for k, v in ACTIVATION.module_dict.items():
    xprint(f"{k}: {v}", color='red')
xprint('', hl='-', hl_style='full')
    
import torch

input = torch.randn(2)

act_cfg = dict(type='Sigmoid')
activation = ACTIVATION.build(act_cfg)
output = activation(input)
# call Sigmoid.forward
print(output)