from onnx import TensorProto
from onnx.helper import (make_model, make_node, make_graph, 
                         make_tensor, make_tensor_value_info)
from onnx.checker import check_model


def shape2tuple(shape):
    return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)


# -------------------------- inputs & outputs --------------------------
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

# -------------------------- nodes & graph --------------------------
node1 = make_node(op_type='MatMul', 
                  inputs=['X', 'A'],
                  outputs=['XA'])

node2 = make_node(op_type='Add', 
                  inputs=['XA', 'B'],
                  outputs=['Y'])

graph = make_graph(nodes=[node1, node2],  # 节点
                   name='lr',  # 名称
                   inputs=[X, A, B],  # 输入节点
                   outputs=[Y])  # 输出节点

# -------------------------- model --------------------------
onnx_model = make_model(graph=graph)
check_model(model=onnx_model)  # 如果测试失败，将引发异常

# 序列化保存模型
save_path = 'ONNX/example_models/linear_regression-serialized.onnx'
with open(save_path, 'wb') as f:
  f.write(onnx_model.SerializeToString())

print(f"Serialized model has saved at {save_path}!")  