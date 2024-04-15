import subprocess


image_size = [32, 64, 128, 256, 512, 640, 768, 1024]
models = []
for _HW in image_size:
    shape_single_batch = f"1x3x{_HW}x{_HW}"  # 输入图片shape
    shape_multi_batch = f"16x3x{_HW}x{_HW}"  # 输入图片shape

    # 定义原始 ONNX 文件和简化后的 ONNX 文件的对应关系
    models.append((f"ONNX/saves/model-fix_dims-{shape_single_batch}.onnx", f"ONNX/saves/model-fix_dims-{shape_single_batch}-sim.onnx"))
    models.append((f"ONNX/saves/model-fix_dims-{shape_multi_batch}.onnx", f"ONNX/saves/model-fix_dims-{shape_multi_batch}-sim.onnx"))

models.append(("ONNX/saves/model-dynamic_dims.onnx", "ONNX/saves/model-dynamic_dims-sim.onnx"))

# 遍历模型列表，执行 onnxsim 命令
for original_model, simplified_model in models:
    # 构建 onnxsim 命令
    command = f"python -m onnxsim {original_model} {simplified_model}"

    # 使用 subprocess 运行命令
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"成功简化模型：{original_model}")
    except subprocess.CalledProcessError as e:
        print(f"简化模型时出错：{e}")
