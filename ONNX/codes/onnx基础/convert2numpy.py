from onnx import TensorProto
from onnx.helper import tensor_dtype_to_np_dtype, \
                        tensor_dtype_to_string


np_dtype = tensor_dtype_to_np_dtype(TensorProto.FLOAT)
print(f"将 ONNX 的 [{tensor_dtype_to_string(TensorProto.FLOAT)}] 数据类型转换为"
      f"Numpy 的 [{np_dtype}] 数据类型")