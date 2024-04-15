import onnx


# 第一种方法
weights_path = 'ONNX/saves/linear_regression.onnx'
onnx_model = onnx.load(weights_path)

# -------------------------- 获取 metadata --------------------------
for field in ['doc_string', 'domain', 'functions',
              'ir_version', 'metadata_props', 'model_version',
              'opset_import', 'producer_name', 'producer_version',
              'training_info']:
    print(field, getattr(onnx_model, field))
    
    
import onnx


weights_path = 'ONNX/saves/linear_regression.onnx'
onnx_model = onnx.load(weights_path)

print(f"[metadata] ir_version: {onnx_model.ir_version}")
for opset in onnx_model.opset_import:
    print(f"[metadata] opset domain={opset.domain!r} version={opset.version!r}")


import onnx


weights_path = 'ONNX/saves/linear_regression.onnx'
onnx_model = onnx.load(weights_path)

# 删除掉目前模型的 opset
del onnx_model.opset_import[:]

# 我们自己定义opset
opset = onnx_model.opset_import.add()
opset.domain = ''
opset.version = 14

print(f"[metadata] ir_version: {onnx_model.ir_version}")
for opset in onnx_model.opset_import:
    print(f"[metadata] opset domain={opset.domain!r} version={opset.version!r}")