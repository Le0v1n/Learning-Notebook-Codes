import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import make_node
from onnx.reference import ReferenceEvaluator


node = make_node('EyeLike', ['X'], ['Y'])

sess = ReferenceEvaluator(node)

x = numpy.random.randn(4, 2).astype(numpy.float32)
feeds = {'X': x}

result = sess.run(None, feeds)

print(f"The node result is: \n{result}\n"
      f"It's type: {type(result)}\n"
      f"Specific type: {type(result[0])}")