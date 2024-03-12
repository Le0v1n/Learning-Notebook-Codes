import os
import sys
import onnx
from typing import Union, List

sys.path.append(os.getcwd())
from utils.outer import xprint


def modify_onnx_axis(weights_path: str, 
                             save_path: Union[str, None], 
                             input_num: Union[int, None] = 1,
                             input_axis: Union[List[List[int]], None] = None,
                             input_axis_name: Union[List[List[str]], None] = None,
                             output_num: Union[int, None] = 2,
                             output_axis: Union[List[List[int]], None] = None,
                             output_axis_name: Union[List[List[str]], None]= None,
                             verbose: bool = False) -> str:
    """
    将ONNX模型的维度修改为动态维度，以便支持不同尺寸的输入。

    - Args:
        - weights_path (str): 要修改的ONNX模型的文件路径。
        - save_path (Union[str, None]): 修改后模型的保存路径。
            - 如果为None，则默认将修改后的模型保存在weights_path所在目录，文件名后添加'modified_'。
        - input_num (Union[int, None]): 要修改的输入数量。默认为1。
            - 如果为None，则不修改任何输入维度。
        - input_axis (Union[List[List[int]], None]): 要修改的输入维度的索引列表。
            - 如果为None，则不修改任何输入维度。
            - 默认为[[0,]]，表示修改第一个维度。
        - input_axis_name (Union[List[List[str]], None]): 修改后的输入维度的名称列表。
            - 如果为None，则不修改任何输入维度名称。
            - 默认为[['B',]]，表示第一个维度名称为'B'。
        - output_num (Union[int, None]): 要修改的输出数量。默认为2。
            - 如果为None，则不修改任何输出维度。
        - output_axis (Union[List[List[int]], None]): 要修改的输出维度的索引列表。
            - 如果为None，则不修改任何输出维度。
            - 默认为[[0, 2, 3], [0, 2, 3]]，表示修改前两个输出的相应维度。
        - output_axis_name (Union[List[List[str]], None]): 修改后的输出维度的名称列表。
            - 如果为None，则不修改任何输出维度名称。
            - 默认为[['B', '512', '512'], ['B', '512', '512']]，表示相应维度名称为'B', '512', '512'。

    - Return:
        - str: 修改后模型的保存路径。

    - 💡  Example:
        - modify_onnx_dynamic_axis(weights_path='save_dir/deeplabv3plus.onnx', 
                                 save_path=None,
                                 input_num=1,
                                 input_axis=[[0]],
                                 input_axis_name=[['B']],
                                 output_num=2,
                                 output_axis=[
                                     [0, 1, 2, 3],
                                     [0, 1, 2, 3],
                                 ],
                                 output_axis_name=[
                                     ['B', '1', '512', '512'],
                                     ['B', '1', '512', '512'],
                                 ])
        
        - 转换完成后:
            - [输入] tensor: float32[batch,3,512,512] --> tensor: float32[B,3,512,512]
            - [输出] tensor: int64[batch,1,ArgMaxlabel_dim_2,ArgMaxlabel_dim_3] --> tensor: int64[B,1,512,512]
            - [输出] float32[batch,1,ArgMaxlabel_dim_2,ArgMaxlabel_dim_3] --> tensor: float32[B,1,512,512]

    - ⚠️  Notes:
        - 输入和输出的维度索引和名称必须是嵌套的列表。
        - 维度名称必须是字符串类型。
        - 如果`save_path`为None，则修改后的模型将保存在与`weights_path`相同的目录下。
    """
    # 函数实现...

    # 加载ONNX模型
    model = onnx.load(weights_path)
    
    # 获取模型的图
    graph = model.graph
    
    # 修改输入的动态轴
    if input_num is not None and input_axis is not None and input_axis_name is not None:
        for i in range(input_num):
            # 获取输入
            input = graph.input[i]
            
            # 获取并修改维度和维度名称
            dim = input.type.tensor_type.shape.dim
            for j, axis in enumerate(input_axis[i]):
                dim[axis].dim_param = input_axis_name[i][j]
    else:
        xprint(f"⚠️  不修改输入", color='yellow') if verbose else ...
    
    # 修改输出的动态轴
    if output_num is not None and output_axis is not None and output_axis_name is not None:
        for i in range(output_num):
            # 获取输出
            output = graph.output[i]
            
            # 获取并修改维度和维度名称
            dim = output.type.tensor_type.shape.dim
            for j, axis in enumerate(output_axis[i]):
                dim[axis].dim_param = output_axis_name[i][j]
    else:
        xprint(f"⚠️  不修改输出", color='yellow') if verbose else ...
    
    # 保存修改后的模型
    if save_path is None:
        pre, ext = os.path.splitext(weights_path)
        save_path = pre + '_modified' + ext
    
    onnx.save(model, save_path)
    
    xprint(f"✔️  ONNX模型修改完成，已保存到 {save_path}!", bold=True, color='green')
    
    return save_path


if __name__ == "__main__":
    path1 = modify_onnx_axis('save_dir/deeplabv3plus_r50-d8_512x512_20k_voc12aug_20200617_102323-aad58ef1.onnx', 
                                     save_path=None,
                                     input_num=1,
                                     input_axis=[[0]],
                                     input_axis_name=[['B']],
                                     output_num=2,
                                     output_axis=[
                                         [0, 1, 2, 3],
                                         [0, 1, 2, 3],
                                     ],
                                     output_axis_name=[
                                         ['B', '1', '512', '512'],
                                         ['B', '1', '512', '512'],
                                     ], verbose=True)

