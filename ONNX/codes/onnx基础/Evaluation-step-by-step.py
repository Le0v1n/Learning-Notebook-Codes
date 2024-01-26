import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info)
from onnx.checker import check_model
from onnx.reference import ReferenceEvaluator


X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])
graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
onnx_model = make_model(graph)
check_model(onnx_model)

for verbose in [1, 2, 3, 4]:
      print()
      print(f"------ verbose={verbose}")
      print()
      sess = ReferenceEvaluator(onnx_model, verbose=verbose)

      x = numpy.random.randn(4, 2).astype(numpy.float32)
      a = numpy.random.randn(2, 1).astype(numpy.float32)
      b = numpy.random.randn(1, 1).astype(numpy.float32)
      feeds = {'X': x, 'A': a, 'B': b}

      result = sess.run(None, feeds)

      print(f"No.{verbose} result is: \n{result}")