import numpy as np
import onnx
from onnx import numpy_helper, TensorProto
from onnx.helper import (make_tensor_value_info, 
                         make_node, make_graph, make_model)
from onnx.checker import check_model


# -------------------------- 创建 initializers --------------------------
value = np.array([0.5, -0.6], dtype=np.float32)
A = numpy_helper.from_array(value, name='A')

value = np.array([0.4], dtype=np.float32)
C = numpy_helper.from_array(value, name='C')

# -------------------------- 创建 输入、输出、节点、图、模型 --------------------------
X = make_tensor_value_info(name='X', elem_type=TensorProto.FLOAT, shape=[None, None])
Y = make_tensor_value_info(name='Y', elem_type=TensorProto.FLOAT, shape=[None])

# 输入是['X', 'A']，输出是['AX']，那么意思就是说，将输入X与参数A相乘，得到输出AX
node1 = make_node(op_type='MatMul', inputs=['X', 'A'], outputs=['AX'])

# 输入是['AX', 'C']，输出是['Y']，那么意思就是说，将输入AX与参数C相加，得到输出Y --> Y <=> AX + C
node2 = make_node(op_type='Add', inputs=['AX', 'C'], outputs=['Y'])

# 创建图的时候输入就是最一开始的输入，输出就是最终的输出
graph = make_graph(nodes=[node1, node2], 
                   name='lr', 
                   inputs=[X], 
                   outputs=[Y], 
                   initializer=[A, C])

# 根据图创建模型
onnx_model = make_model(graph=graph)
check_model(onnx_model)  # 检查模型

# -------------------------- 查看初始化器 --------------------------
print(f" -------------------------- 查看初始化器 --------------------------")
for init in onnx_model.graph.initializer:
    print(init)