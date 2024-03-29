import torch
from torchvision import models
import onnx
import argparse


def parse_list(s):
    try:
        return list(map(int, s.strip('[]').split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError('Invalid list format. Must be comma-separated integers.')


# ==================================== 参数 ==================================== 
parser = argparse.ArgumentParser()
parser.add_argument('--input-shape', type=parse_list, default=[1, 3, 640, 640], help='The shape of input')
parser.add_argument('--device', type=str, default='cpu', help='The shape of input')
parser.add_argument('--verbose', action='store_true', help='')
args = parser.parse_args()  # 解析命令行参数

_shape = "x".join(map(str, args.input_shape))
onnx_save_path = f'ONNX/saves/model-fix_dims-{_shape}.onnx'  # 导出的ONNX模型路径 
# ==============================================================================

# 创建一个训练好的模型
model = models.mobilenet_v3_small(pretrained=True)  # ImageNet 预训练权重
model = model.eval().to(args.device)

# 构建一个输入
dummy_input = torch.randn(size=args.input_shape).to(args.device)  # [N, B, H, W]

# 让模型推理
# output = model(dummy_input)
# print(f"output.shape: {output.shape}\n")

# ------ 使用 PyTorch 自带的函数将模型转换为 ONNX 格式
with torch.no_grad():
    torch.onnx.export(
        model=model,                            # 要转换的模型
        args=dummy_input,                       # 模型的输入
        f=onnx_save_path,                       # 导出的ONNX模型路径 
        input_names=['input'],                  # ONNX模型输入的名字(自定义)
        output_names=['output'],                # ONNX模型输出的名字(自定义)
        opset_version=17,                       # Opset算子集合的版本（默认为17）
    )
    
print(f"ONNX 模型导出成功，路径为：{onnx_save_path}\n")

# ------ 验证导出的模型是否正确
# 读取导出的模型
onnx_model = onnx.load(onnx_save_path)

# 检查模型是否正常
onnx.checker.check_model(onnx_model)

print(f"{_shape}模型已导出!")