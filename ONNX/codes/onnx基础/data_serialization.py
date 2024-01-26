import numpy as np
from onnx.numpy_helper import from_array


# 创建一个 numpy 的 Tensor
numpy_tensor = np.array([0, 1, 4, 5, 3], dtype=np.float32)
print(type(numpy_tensor))

# 创建一个 onnx 的 Tensor
onnx_tensor = from_array(numpy_tensor)
print(type(onnx_tensor))

# 将 onnx 的 Tensor 序列化
serialized_tensor = onnx_tensor.SerializeToString()
print(type(serialized_tensor))

# 将序列化的 onnx Tensor 保存到本地
save_path = 'ONNX/example_models/saved_serialized_tensor.pb'  # pb: Protocol Buffers 
with open(save_path, 'wb') as f:
  f.write(serialized_tensor)
print(f"The serialized onnx tensor has been saved at {save_path}!")