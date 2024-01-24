import onnx
from onnx import TensorProto
from onnx.helper import (make_model, make_node, make_graph, 
                         make_tensor, make_tensor_value_info)
from onnx.checker import check_model


# -------------------------- inputs --------------------------
# 'X'是名称，TensorProto.FLOAT是类型，[None, None]是形状。
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])

# -------------------------- outputs(形状未定义) --------------------------
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

# -------------------------- nodes --------------------------
# 它创建一个由运算符类型MatMul定义的节点，'X'、'A'是节点的输入，'XA'是输出。
node1 = make_node(op_type='MatMul', 
                  inputs=['X', 'A'],
                  outputs=['XA'])

node2 = make_node(op_type='Add', 
                  inputs=['XA', 'B'],
                  outputs=['Y'])

# -------------------------- graph --------------------------
# 从节点到图，图是由节点列表、输入列表、输出列表和名称构建的。
graph = make_graph(nodes=[node1, node2],  # 节点
                   name='lr',  # 名称
                   inputs=[X, A, B],  # 输入节点
                   outputs=[Y])  # 输出节点

# -------------------------- model --------------------------
# ONNX图，这种情况下没有元数据。
onnx_model = make_model(graph=graph)

# 让我们检查模型是否一致，这个函数在“Checker and Shape Inference”部分有描述。
check_model(model=onnx_model)  # 如果测试失败，将引发异常

print(onnx_model)

# 将这个模型保存到本地
onnx.save_model(onnx_model, 'ONNX/example_models/linear_regression.onnx')