import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, AttributeProto
from onnx.helper import (make_tensor_value_info, make_tensor, make_function, 
                         make_node, make_graph, make_model, set_model_props,
                         make_opsetid)
from onnx.checker import check_model


# -------------------------- 定义一个线性回归的函数 --------------------------
# 新的领域名称
new_domain = 'custom_domain'

# 构建 opset_imports 列表，包含两个 OpsetID，分别为默认领域和自定义领域
opset_imports = [
    make_opsetid(domain="", version=14),
    make_opsetid(domain=new_domain, version=1)
]

# 第一步是创建一个与函数的输入参数相等的常数
cst = make_node(op_type='Constant', inputs=[], outputs=['B'])

att = AttributeProto()
att.name = 'value'

# 这行代码指示该值来自函数所给定的名为 'bias' 的参数
att.ref_attr_name = 'bias'
att.type = AttributeProto.TENSOR
cst.attribute.append(att)

node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

linear_regression = make_function(
    domain=new_domain,  # 作用域名称（指定函数的作用域名称）
    fname='LinearRegression',  # 函数名称（指定函数的名称）
    inputs=['X', 'A'],  # 输入的名称（定义函数的输入张量的名称列表）
    outputs=['Y'],  # 输出的名称（定义函数的输出张量的名称列表）
    nodes=[cst, node1, node2],  # 使用到的节点（定义函数使用到的节点列表）
    opset_imports=opset_imports,  # opset（指定 OpsetID 列表，定义函数使用的运算符版本）
    attributes=[],  # 属性的名称（定义函数的属性列表）
)

# -------------------------- 定义图 --------------------------
X = make_tensor_value_info(name='X', elem_type=TensorProto.FLOAT, shape=[None, None])
A = make_tensor_value_info(name='A', elem_type=TensorProto.FLOAT, shape=[None, None])
B = make_tensor_value_info(name='B', elem_type=TensorProto.FLOAT, shape=[None, None])
Y = make_tensor_value_info(name='Y', elem_type=TensorProto.FLOAT, shape=[None])

graph = make_graph(
    nodes=[make_node(op_type='LinearRegression', 
                     inputs=['X', 'A'], outputs=['Y1'], 
                     domain=new_domain, bias=make_tensor('former_B', TensorProto.FLOAT, 
                                                         dims=[1], vals=[0.67])),
           make_node(op_type='Abs', inputs=['Y1'], outputs=['Y'])],
    name='example',
    inputs=[X, A],
    outputs=[Y]
)

# -------------------------- 定义模型 --------------------------
onnx_model = make_model(graph=graph, 
                        opset_imports=opset_imports,
                        functions=[linear_regression])
check_model(onnx_model)

print(onnx_model)

model_save_path = 'ONNX/saves/function-with_attribute.onnx'
onnx.save(onnx_model, model_save_path)