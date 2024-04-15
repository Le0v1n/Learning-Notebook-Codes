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

onnx_fix         = [0.0575, 0.0585, 0.0611, 0.0627, 0.0714, 0.0776, 0.0867, 0.1103, 0.2410, 0.2443, 
                    0.2618, 0.3097, 0.5556, 0.7191, 0.9293, 1.4768, ]
# onnx_fix_sim     = [0.0597, 0.0593, 0.0686, 0.0811, 0.1241, 0.1525, 0.1946, 0.2957, 0.2545, 0.2745, 
#                     0.3181, 0.7112, 1.2586, 1.6429, 2.2830, 3.9742,]
onnx_dynamic     = [0.0619, 0.0591, 0.0597, 0.0622, 0.0703, 0.0785, 0.0861, 0.1126, 0.2295, 0.2421, 
                    0.2576, 0.3131, 0.5873, 0.7130, 0.9285, 1.4945, ]
# onnx_dynamic_sim = [0.0585, 0.0621, 0.0694, 0.0789, 0.1256, 0.1579, 0.2038, 0.3045, 0.2498, 0.2655, 
#                     0.3544, 0.6122, 1.1949, 1.6693, 2.3303, 3.9104,]
pytorch_CPU      = [0.0636, 0.0643, 0.0629, 0.0690, 0.0841, 0.0975, 0.1138, 0.1630, 0.2538, 0.2576, 
                    0.2804, 0.3502, 0.7655, 0.8988, 1.5091, 3.3530, ]
pytorch_GPU      = [0.0731, 0.0701, 0.0700, 0.0731, 0.0765, 0.0823, 0.0851, 0.0958, 0.2446, 0.2481, 
                    0.2727, 0.3043, 0.3970, 0.4877, 0.5754, 1.1316, ]

exps = ['Full-Data', 'Single-Batch', 'Multi-Batch', 'Sim']
for exp_name in exps:
    if exp_name == 'Full-Data':
        methods_use = methods
        onnx_fix_use = onnx_fix
        # onnx_fix_sim_use = onnx_fix_sim
        onnx_dynamic_use = onnx_dynamic
        # onnx_dynamic_sim_use = onnx_dynamic_sim
        pytorch_CPU_use = pytorch_CPU
        pytorch_GPU_use = pytorch_GPU
    elif exp_name == 'Single-Batch':
        methods_use = methods[:8]
        onnx_fix_use = onnx_fix[:8]
        # onnx_fix_sim_use = onnx_fix_sim[:8]
        onnx_dynamic_use = onnx_dynamic[:8]
        # onnx_dynamic_sim_use = onnx_dynamic_sim[:8]
        pytorch_CPU_use = pytorch_CPU[:8]
        pytorch_GPU_use = pytorch_GPU[:8]
    elif exp_name == 'Multi-Batch':
        methods_use = methods[8:]
        onnx_fix_use = onnx_fix[8:]
        # onnx_fix_sim_use = onnx_fix_sim[8:]
        onnx_dynamic_use = onnx_dynamic[8:]
        # onnx_dynamic_sim_use = onnx_dynamic_sim[8:]
        pytorch_CPU_use = pytorch_CPU[8:]
        pytorch_GPU_use = pytorch_GPU[8:]
    elif exp_name == 'Sim':
        methods_use = methods[8:]
        # onnx_fix_sim_use = onnx_fix_sim[8:]
        # onnx_dynamic_sim_use = onnx_dynamic_sim[8:]
    else:
        raise KeyError()
    
    plt.figure(figsize=(10, 6), dpi=200)

    # 绘制折线图
    plt.plot(methods_use, onnx_fix_use, marker='o', label='ONNX-fix', linestyle='-', markersize=5) if exp_name != 'Sim' else ...
    # plt.plot(methods_use, onnx_fix_sim_use, marker='o', label='ONNX-fix-sim', linestyle='-', markersize=5)
    plt.plot(methods_use, onnx_dynamic_use, marker='o', label='ONNX-dynamic', linestyle='--', markersize=5) if exp_name != 'Sim' else ...
    # plt.plot(methods_use, onnx_dynamic_sim_use, marker='o', label='ONNX-dynamic-sim', linestyle='--', markersize=5)
    plt.plot(methods_use, pytorch_CPU_use, marker='o', label='PyTorch-CPU', linestyle='-.', markersize=5) if exp_name != 'Sim' else ...
    plt.plot(methods_use, pytorch_GPU_use, marker='o', label='PyTorch-GPU', linestyle='-.', markersize=5) if exp_name != 'Sim' else ...

    # 设置图表标题和标签
    plt.title(f'Speed Comparison of Different Models ({exp_name})')
    plt.xlabel('Input Shapes')
    plt.ylabel('Average cost time (s)')
    plt.xticks(rotation=30, ha='right')
    plt.grid(True, alpha=0.3, linestyle='-')

    # 显示图表
    # plt.tight_layout()
    plt.legend()
    plt.savefig(f'ONNX/saves/Speed-Comparison-of-Different-Models-{exp_name}.jpg')
    plt.show()
