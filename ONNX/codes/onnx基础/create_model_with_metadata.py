import onnx


# -------------------------- 加载模型 --------------------------
weights_path = 'ONNX/saves/linear_regression.onnx'
onnx_model = onnx.load(weights_path)

# -------------------------- 修改metadata --------------------------
onnx_model.model_version = 15
onnx_model.producer_name = 'Le0v1n'
onnx_model.producer_version = 'v1.0'
onnx_model.doc_string = 'documentation about this onnx model by Le0v1n'

# 读取模型现在的metadata属性
prop = onnx_model.metadata_props
print(prop)  # []

# 目前 metadata属性中的内容为空，我们可以往里面放一些信息
# ⚠️ metadata_props只接受字典
info1 = {'model说明': '这是一个用于学习的ONNX模型', 
         '时间': '20240123'}
onnx.helper.set_model_props(onnx_model, info1)
print(onnx_model)
