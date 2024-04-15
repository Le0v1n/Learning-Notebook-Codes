import torch
import sys
sys.path.append('/mnt/f/Projects/本地代码/Learning-Notebook-Codes/ObjectDetection/YOLOv5/codes/yolov5-v7.0')
from thop import profile
from models.common import SPP, SPPF


spp = SPP(c1=1024, c2=1024)
sppf = SPPF(c1=1024, c2=1024)

# 定义输入大小
input_data = torch.randn(1, 1024, 20, 20)

# 使用 thop 的 profile 函数获取 FLOPs 和参数数量
spp_flops, spp_params = profile(spp, inputs=(input_data,))
sppf_flops, sppf_params = profile(sppf, inputs=(input_data,))

print(f"SPP FLOPs: {spp_flops / 1e9} G FLOPs")
print(f"SPP Params: {spp_params / 1e6} M Params")
print(f"SPPF FLOPs: {sppf_flops / 1e9} G FLOPs")
print(f"SPPF Params: {sppf_params / 1e6} M Params")
