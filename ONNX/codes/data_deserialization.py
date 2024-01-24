from onnx import TensorProto
from onnx.numpy_helper import to_array


# 读取序列化数据
data_path = 'ONNX/example_models/saved_serialized_tensor.pb'  # pb: Protocol Buffers 
with open(data_path, 'rb') as f:
  serialized_tensor = f.read()
print(f"--------------------------- serialized_tensor ---------------------------\n"
      f"{type(serialized_tensor)}\n"  # <class 'bytes'>
      f"{serialized_tensor}\n")

"""
我们发现此时 serialized_tensor 的数据类型并不是我们想要的 onnx.onnx_ml_pb2.TensorProto
而是 <class 'bytes'>，所以我们需要将其转换为 onnx.onnx_ml_pb2.TensorProto 格式
"""
# 创建一个空的 onnx tensor
onnx_tensor = TensorProto()

# 从二进制字符串 serialized_tensor 中解析数据，并将解析后的结果存储在 onnx_tensor 对象中
onnx_tensor.ParseFromString(serialized_tensor)
print(f"--------------------------- onnx_tensor ---------------------------\n"
      f"{type(onnx_tensor)}\n"
      f"{onnx_tensor}\n")

# 将 onnx 的 Tensor 转换为 numpy 的Tensor
numpy_tensor = to_array(onnx_tensor)
print(f"--------------------------- numpy_tensor ---------------------------\n"
      f"{type(numpy_tensor)}\n"
      f"{numpy_tensor}")
