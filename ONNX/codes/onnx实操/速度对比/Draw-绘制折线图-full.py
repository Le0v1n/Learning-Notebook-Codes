import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


config = {
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman']
}
rcParams.update(config)

# 模型名称和性能数据
methods = ['[1, 3, 32, 32]', '[1, 3, 64, 64]', '[1, 3, 128, 128]', '[1, 3, 256, 256]', '[1, 3, 512, 512]', 
           '[1, 3, 640, 640]', '[1, 3, 768, 768]', '[1, 3, 1024, 1024]', 
           '[18, 3, 32, 32]', '[18, 3, 64, 64]', '[18, 3, 128, 128]', '[18, 3, 256, 256]', '[18, 3, 512, 512]', 
           '[18, 3, 640, 640]', '[18, 3, 768, 768]', '[18, 3, 1024, 1024]']

onnx_fix         = ['0.0658', '0.0683', '0.0747', '0.0893', '0.1484', '0.1983', '0.2529', '0.3888', '0.3252', '0.3468', 
                    '0.4244', '0.6910', '1.7164', '2.4357', '3.5806', '6.0836']
onnx_fix_sim     = ['0.0679', '0.0701', '0.0728', '0.0901', '0.1486', '0.1947', '0.2488', '0.3959', '0.3255', '0.3509', 
                    '0.4295', '0.6859', '1.7125', '2.4594', '3.5368', '6.1163']
onnx_dynamic     = ['0.0669', '0.0684', '0.0732', '0.0883', '0.1544', '0.1946', '0.2535', '0.4008', '0.3268', '0.3504', 
                    '0.4297', '0.7005', '1.7420', '2.4750', '3.6693', '6.2694']
onnx_dynamic_sim = ['0.0667', '0.0694', '0.0755', '0.0901', '0.1485', '0.1935', '0.2555', '0.3996', '0.3262', 
                    '0.3554', '0.4269', '0.7020', '1.7531', '2.5205', '3.6110', '6.3470']
pytorch          = ['0.0740', '0.0734', '0.0784', '0.1070', '0.1906', '0.2561', '0.3303', '0.5216', '0.3357', '0.3653', 
                    '0.4806', '0.8770', '3.6240', '4.3787', '10.3582', '10.3582']

onnx_fix = [float(val) for val in onnx_fix]
onnx_fix_sim = [float(val) for val in onnx_fix_sim]
onnx_dynamic = [float(val) for val in onnx_dynamic]
onnx_dynamic_sim = [float(val) for val in onnx_dynamic_sim]
pytorch = [float(val) for val in pytorch]

# 创建一个新的图表
plt.figure(figsize=(10, 6), dpi=200)
# plt.figure(dpi=200)

# 绘制折线图
plt.plot(methods, onnx_fix, marker='o', label='onnx-fix_axis', linestyle='-', markersize=5)
plt.plot(methods, onnx_fix_sim, marker='o', label='onnx-fix_axis-simplified', linestyle='-', markersize=5)
plt.plot(methods, onnx_dynamic, marker='o', label='onnx-dynamic_axis', linestyle='--', markersize=5)
plt.plot(methods, onnx_dynamic_sim, marker='o', label='onnx-dynamic_axis-simplified', linestyle='--', markersize=5)
plt.plot(methods, pytorch, marker='o', label='pytorch', linestyle='-.', markersize=5)

# # # 设置y轴刻度的间隔
# custom_ticks = [0, 0.01, 0.03, 0.5, 1.0, 5.0, 10]
# plt.yticks(custom_ticks)

# 设置图表标题和标签
plt.title('Speed Comparison of Different Models on CPU')
plt.xlabel('Input Shapes')
plt.ylabel('Cost time (s)')
plt.xticks(rotation=30, ha='right')
plt.ylim(0, 6.2)
# plt.grid(True, alpha=0.3, linestyle='-')

# 显示图表
# plt.tight_layout()
plt.legend()
plt.savefig('ONNX/saves/speed_comparison.jpg')
plt.show()
