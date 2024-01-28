import subprocess

image_size = [32, 64, 128, 256, 512, 640, 768, 1024]
test_times = 50  # 测试次数
warm_up = 'no'  # 是否开启热身

for _HW in image_size:
    shape = f"[1,3,{_HW},{_HW}]"  # 输入图片shape
    subprocess.run([
        "python",
        "ONNX/codes/onnx实操/速度对比/No1-不同尺度下单张图片推理/ONNX-fix_dim-single_batch-without_simplified.py",
        f"--input-shape={shape}",
        f"--test-times={test_times}",
        f"--warm-up={warm_up}"
    ])
    # subprocess.run([
    #     "python",
    #     "ONNX/codes/onnx实操/速度对比/No1-不同尺度下单张图片推理/ONNX-fix_dim-single_batch-simplified.py",
    #     f"--input-shape={shape}",
    #     f"--test-times={test_times}",
    #     f"--warm-up={warm_up}"
    # ])
    subprocess.run([
        "python",
        "ONNX/codes/onnx实操/速度对比/No1-不同尺度下单张图片推理/ONNX-dynamic_dim-single_batch-without_simplified.py",
        f"--input-shape={shape}",
        f"--test-times={test_times}",
        f"--warm-up={warm_up}"
    ])
    # subprocess.run([
    #     "python",
    #     "ONNX/codes/onnx实操/速度对比/No1-不同尺度下单张图片推理/ONNX-dynamic_dim-single_batch-simplified.py",
    #     f"--input-shape={shape}",
    #     f"--test-times={test_times}",
    #     f"--warm-up={warm_up}"
    # ])
    subprocess.run([
        "python",
        "ONNX/codes/onnx实操/速度对比/No1-不同尺度下单张图片推理/PyTorch-dynamic_dim_batch-cpu.py",
        f"--input-shape={shape}",
        f"--test-times={test_times}",
        f"--warm-up={warm_up}"
    ])
    subprocess.run([
        "python",
        "ONNX/codes/onnx实操/速度对比/No1-不同尺度下单张图片推理/PyTorch-dynamic_dim_batch-gpu.py",
        f"--input-shape={shape}",
        f"--test-times={test_times}",
        f"--warm-up={warm_up}"
    ])
