import onnx
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--output-path', required=True)
    return parser.parse_args()


def show_inp_and_oup_info(model, modify=False):
    print("\033[1;31m==============================================修改前==============================================\033[0m")
    input_info = model.graph.input
    print("模型的输入信息：")
    for info in input_info:
        print(info.name, info.type)

    print("\033[1;31m--------------------------------------------------------------------------------------------------\033[0m")

    output_info = model.graph.output
    print("模型的输出信息：")
    for info in output_info:
        print(info.name, info.type)
        
    
if __name__ == "__main__":
    args = parse_arguments()

    # 读取 ONNX 模型
    model = onnx.load(args.model_path)
    
    show_inp_and_oup_info(model, modify=False)

    # 找到输入张量并修改
    for input_info in model.graph.input:
        if input_info.name == "x":
            # 修改输入张量的形状
            input_info.type.tensor_type.shape.dim[0].dim_param = "B"

    # 修改输出张量的形状
    for output_info in model.graph.output:
        if output_info.name in ["label", "score"]:
            output_info.type.tensor_type.shape.dim[0].dim_param = "B"
            # output_info.type.tensor_type.shape.dim[2].dim_value = 512
            # output_info.type.tensor_type.shape.dim[3].dim_value = 512
            
    show_inp_and_oup_info(model, modify=True)
    
    # 保存修改后的模型
    onnx.save(model, args.output_path)