import onnx
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

# -------------------------- Check: Inputs --------------------------
print(f"-------------------------- inputs --------------------------")
# print(onnx_model.graph.input)
"""
[name: "X"      
type {
  tensor_type { 
    elem_type: 1
    shape {     
      dim {     
      }
      dim {     
      }
    }
  }
}
, name: "A"     
type {
  tensor_type { 
    elem_type: 1
    shape {
      dim {
      }
      dim {
      }
    }
  }
}
, name: "B"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
      }
      dim {
      }
    }
  }
}
]
"""

for obj in onnx_model.graph.input:
    print(f"name={obj.name!r} "
          f"dtype={obj.type.tensor_type.elem_type!r} "
          f"shape={shape2tuple(obj.type.tensor_type.shape)!r}")
    
# -------------------------- Check: Outputs --------------------------
print(f"------------------------- outputs -------------------------")
for obj in onnx_model.graph.output:
    print(f"name={obj.name!r} "
          f"dtype={obj.type.tensor_type.elem_type!r} "
          f"shape={shape2tuple(obj.type.tensor_type.shape)!r}")

# -------------------------- Check: Nodes --------------------------
print(f"-------------------------- nodes --------------------------")
for node in onnx_model.graph.node:
    print(f"name={node.name!r} "
          f"type={node.op_type!r} "
          f"input={node.input!r} "
          f"output={node.output!r}")
    