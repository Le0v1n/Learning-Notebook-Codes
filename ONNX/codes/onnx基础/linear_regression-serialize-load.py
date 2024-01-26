from onnx import load


weights_path = 'ONNX/example_models/linear_regression-serialized.onnx'
with open(weights_path, 'rb') as f:
  onnx_model = load(f)
  
print(onnx_model)