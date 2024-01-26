import onnx


# 读取导出的模型
onnx_path = 'ONNX/saves/resnet18_imagenet.onnx'  # 导出的ONNX模型路径
onnx_model = onnx.load(onnx_path)

# 检查模型是否正常
onnx.checker.check_model(onnx_model)

print(f"模型导出正常!")