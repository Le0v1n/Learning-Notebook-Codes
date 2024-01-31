import torch
import torch.nn as nn
import os
import sys
import platform
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from models.common import Conv


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        _concat = torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1)
        _conv = self.conv(_concat)
        
        print(f"{_concat.shape = }")
        print(f"{_conv.shape = }")
        
        return _conv
        # return self.conv(self.contract(x))
    
    
if __name__ == "__main__":
    # 创建tensor
    input_tensor = torch.randn(size=[1, 3, 256, 256], dtype=torch.float32)
    print(f"{input_tensor.shape = }")
    
    # 创建Focus子模块模型对象
    Sub_module = Focus(3, 64).eval()

    # 前向推理
    output = Sub_module(input_tensor)