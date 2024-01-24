import onnx
from onnx import numpy_helper, TensorProto
from onnx.helper import (make_tensor_value_info, 
                         make_node, make_graph, make_model)
from onnx.checker import check_model


# -------------------------- 不变 --------------------------
X = make_tensor_value_info(name='X', elem_type=TensorProto.FLOAT, shape=[None, None])
A = make_tensor_value_info(name='A', elem_type=TensorProto.FLOAT, shape=[None, None])
B = make_tensor_value_info(name='B', elem_type=TensorProto.FLOAT, shape=[None, None])
Y = make_tensor_value_info(name='Y', elem_type=TensorProto.FLOAT, shape=[None])

# -------------------------- 新算子：transpose --------------------------
node_transpose = make_node(op_type='Transpose', inputs=['A'], outputs=['tA'], perm=[1, 0])

# -------------------------- 创建 输入、输出、节点、图、模型 --------------------------
node1 = make_node(op_type='MatMul', inputs=['X', 'tA'], outputs=['XA'])
node2 = make_node(op_type='Add', inputs=['XA', 'B'], outputs=['Y'])

graph = make_graph(nodes=[node_transpose, node1, node2], 
                   name='example', 
                   inputs=[X, A, B], 
                   outputs=[Y])

# 根据图创建模型
onnx_model = make_model(graph=graph)
check_model(onnx_model)  # 检查模型

model_save_path = 'ONNX/saves/attributes-transpose.onnx'
onnx.save(onnx_model, model_save_path)
print(f"ONNX model with initializer has been saved to {model_save_path}")
print(onnx_model)