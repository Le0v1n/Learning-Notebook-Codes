import numpy as np
import onnx
from onnx.helper import make_tensor_value_info, make_node, make_graph, make_model
from onnx.numpy_helper import from_array
from onnx.checker import check_model
from onnxruntime import InferenceSession

# -------------------------- 初始化器 --------------------------
# 创建一个包含值为0的浮点数数组，并指定数据类型为np.float32
value = np.array([0], dtype=np.float32)

# 使用onnx.numpy_helper.from_array将numpy数组转换为ONNX的TensorProto形式
zero = from_array(value, name='zero')

# -------------------------- 输入 --------------------------
# 创建输入Tensor信息，名称为'X'，数据类型为onnx.TensorProto.FLOAT，形状为[None, None]，表示可变维度
X = make_tensor_value_info('X', onnx.TensorProto.FLOAT, shape=[None, None])

# 创建输出Tensor信息，名称为'Y'，数据类型为onnx.TensorProto.FLOAT，形状为[None]，表示可变维度
Y = make_tensor_value_info('Y', onnx.TensorProto.FLOAT, shape=[None])

# -------------------------- 节点 --------------------------
# 创建 ReduceSum 节点，用于沿着指定轴对输入Tensor进行求和，输入为 'X'，输出为 'rsum'
rsum = make_node(op_type='ReduceSum', inputs=['X'], outputs=['rsum'])

# 创建 Greater 节点，用于比较 'rsum' 和 'zero'，输出结果保存在 'cond'
cond = make_node(op_type='Greater', inputs=['rsum', 'zero'], outputs=['cond'])

# -------------------------- 图形（带有条件） --------------------------
"""
    then <=> True:  表示当条件满足的时候执行的
    else <=> False: 表示当条件不满足的时候执行的
"""
# -------------------------- 图形: True -> then --------------------------
# 条件为True时的输出Tensor信息
then_out = make_tensor_value_info(name='then_out', 
                                  elem_type=onnx.TensorProto.FLOAT, 
                                  shape=None)

# 用于返回的常量Tensor
then_cst = from_array(np.array([1]).astype(np.float32))

# 创建 Constant 节点，将常量Tensor作为输出 'then_out' 的值，构成一个单一节点
then_const_node = make_node(op_type='Constant', 
                            inputs=[], 
                            outputs=['then_out'], 
                            value=then_cst, 
                            name='cst1')

# 创建包裹这些元素的图形，表示当条件为真时执行
then_body = make_graph(nodes=[then_const_node], 
                       name='then_body', 
                       inputs=[], 
                       outputs=[then_out])

# -------------------------- 图形: False -> else --------------------------
# 对于 else 分支，相同的处理过程
else_out = make_tensor_value_info(name='else_out', 
                                  elem_type=onnx.TensorProto.FLOAT, 
                                  shape=[5])

else_cst = from_array(np.array([-1]).astype(np.float32))

else_const_node = make_node(op_type='Constant', 
                            inputs=[], 
                            outputs=['else_out'], 
                            value=else_cst, 
                            name='cst2')

else_body = make_graph(nodes=[else_const_node], name='else_body', inputs=[], outputs=[else_out])

# 创建 If 节点，接受条件 'cond'，并有两个分支，分别为 'then_body' 和 'else_body'。
if_node = make_node(op_type='If', inputs=['cond'], outputs=['Y'], 
                    then_branch=then_body, 
                    else_branch=else_body)

# 创建整体的图形，包括 ReduceSum、Greater 和 If 节点
graph = make_graph(nodes=[rsum, cond, if_node],
                   name='if',
                   inputs=[X],
                   outputs=[Y],
                   initializer=[zero])

# -------------------------- 模型 --------------------------
# 创建 ONNX 模型，使用之前构建的图形作为参数
onnx_model = make_model(graph=graph)

# 检查模型的有效性，确保模型结构符合 ONNX 规范
check_model(onnx_model)

# 删除原有的 opset
del onnx_model.opset_import[:]

# 添加新的 opset
opset = onnx_model.opset_import.add()
opset.domain = ''
opset.version = 15

# 设置 ONNX 模型的 IR 版本和文档字符串
onnx_model.ir_version = 8
onnx_model.doc_string = '这是一个涉及到 if-else 语句的 ONNX 模型'

# 保存模型
model_save_path = 'ONNX/saves/if-else.onnx'
onnx.save(onnx_model, model_save_path)

print(onnx_model)

# -------------------------- 推理 --------------------------
# 创建推理会话，加载保存的 ONNX 模型
session = InferenceSession(path_or_bytes=model_save_path, 
                           providers=['CPUExecutionProvider'])

# 创建输入张量，全为1，形状为[3, 2]，数据类型为np.float32
input_tensor = np.ones(shape=[3, 2], dtype=np.float32)

# 运行推理，获取输出张量
output_tensor = session.run(output_names=None, 
                            input_feed={'X': input_tensor})

# 打印输出张量
print(f"output: {output_tensor}")
