from onnx import load_tensor_from_string
from onnx.numpy_helper import to_array


# 读取序列化数据
data_path = 'ONNX/example_models/saved_serialized_tensor.pb'  # pb: Protocol Buffers 
with open(data_path, 'rb') as f:
  serialized_tensor = f.read()
print(f"--------------------------- serialized_tensor ---------------------------\n"
      f"{type(serialized_tensor)}\n"  # <class 'bytes'>
      f"{serialized_tensor}\n")

# 更加便捷地加载序列化数据
onnx_tensor = load_tensor_from_string(serialized_tensor)
print(f"--------------------------- onnx_tensor ---------------------------\n"
      f"{type(onnx_tensor)}\n"
      f"{onnx_tensor}\n")

# 将 onnx 的 Tensor 转换为 numpy 的Tensor
numpy_tensor = to_array(onnx_tensor)
print(f"--------------------------- numpy_tensor ---------------------------\n"
      f"{type(numpy_tensor)}\n"
      f"{numpy_tensor}")
