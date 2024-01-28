import subprocess

image_size = [32, 64, 128, 256, 512, 640, 768, 1024]

subprocess.run([
    "python",
    "ONNX/codes/onnx实操/速度对比/B-创建动态维度模型.py"
])

for _HW in image_size:
    shape = f"[1,3,{_HW},{_HW}]"  # 输入图片shape

    subprocess.run([
        "python",
        "ONNX/codes/onnx实操/速度对比/A-创建固定维度模型.py",
        f"--input-shape={shape}",
        "--device=cpu"
    ])
