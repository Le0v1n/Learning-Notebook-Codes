import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/main/onnx/onnx.proto


# Create one input (ValueInfoProto)
X = helper.make_tensor_value_info(name='X',
                                  elem_type=TensorProto.FLOAT, 
                                  shape=[3, 2])
pads = helper.make_tensor_value_info(name='pads', 
                                     elem_type=TensorProto.FLOAT, 
                                     shape=[1, 4])

value = helper.make_tensor_value_info(name='value', 
                                      elem_type=AttributeProto.FLOAT, 
                                      shape=[1])


# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info(name='Y', 
                                  elem_type=TensorProto.FLOAT, 
                                  shape=[3, 4])

# Create a node (NodeProto) - This is based on Pad-11
node_def = helper.make_node(
    op_type='Pad', # node name
    inputs=['X', 'pads', 'value'], # inputs
    outputs=['Y'], # outputs
    mode='constant', # attributes
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    nodes=[node_def],
    name='test-model',
    inputs=[X, pads, value],
    outputs=[Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph=graph_def, producer_name='onnx-example')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')