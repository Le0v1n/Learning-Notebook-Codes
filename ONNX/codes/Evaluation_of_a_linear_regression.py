import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info)
from onnx.checker import check_model
from onnx.reference import ReferenceEvaluator


# -------------------------- 不变 --------------------------
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])

onnx_model = make_model(graph)
check_model(onnx_model)

# -------------------------- 模型评估 --------------------------
# 创建 ReferenceEvaluator 对象，用于运行 ONNX 模型
sess = ReferenceEvaluator(onnx_model)

# 生成随机输入数据
x = numpy.random.randn(4, 2).astype(numpy.float32)
a = numpy.random.randn(2, 1).astype(numpy.float32)
b = numpy.random.randn(1, 1).astype(numpy.float32)

# 将输入数据放入字典中
feeds = {'X': x, 'A': a, 'B': b}

# 使用 ReferenceEvaluator 对象运行模型，获取输出结果
result = sess.run(None, feeds)

print(f"The model result is: \n{result}\n"
      f"It's type: {type(result)}\n"
      f"Specific type: {type(result[0])}")