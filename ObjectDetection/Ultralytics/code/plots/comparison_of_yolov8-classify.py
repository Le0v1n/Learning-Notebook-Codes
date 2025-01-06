import os
import matplotlib.pyplot as plt
from matplotlib import rcParams


config = {
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
    "font.size": 14
}

rcParams.update(config)


# ============================== 参数 ==============================
save_dir = 'ObjectDetection/YOLOv8/imgs_markdown'
image_filename = 'Comparison_of_YOLOv8-classify'
filetype = '.png'
figsize = (20, 8)
dpi = 200
first_offset = (5, -5)
offset = (-20, -20)
# ==================================================================

# ---------- DATA ----------
models = ['v8n-cls', 'v8s-cls', 'v8m-cls', 'v8l-cls', 'v8x-cls']
acc_top1 = [69.0, 73.8, 76.8, 76.8, 79.0]
acc_top5 = [88.3, 91.7, 93.5, 93.5, 94.6]
cpu_speed_onnx = [12.9, 23.4, 85.4, 163.0, 232.0]
a100_speed_trt = [0.31, 0.35, 0.62, 0.87, 1.01]
params = [2.7, 6.4, 17.0, 37.5, 57.4]
flops = [4.3, 13.5, 42.7, 99.7, 154.8]

# 创建画布
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize, dpi=dpi)

# 设置颜色循环
colors = plt.cm.tab10.colors

# 绘制折线
axes[0, 0].plot(models, acc_top1, color='#B0C4DE')
axes[0, 1].plot(models, acc_top5, color='#B0C4DE')
axes[0, 2].plot(models, cpu_speed_onnx, color='#B0C4DE')
axes[1, 0].plot(models, a100_speed_trt, color='#B0C4DE')
axes[1, 1].plot(models, params, color='#B0C4DE')
axes[1, 2].plot(models, flops, color='#B0C4DE')

# 绘制点
for i, model in enumerate(models):
    if i == 0:
        xytext = first_offset
    else:
        xytext = offset
    
    axes[0, 0].plot([model], [acc_top1[i]], color=colors[i], marker='o', label=model)
    axes[0, 1].plot([model], [acc_top5[i]], color=colors[i], marker='o', label=model)
    axes[0, 2].plot([model], [cpu_speed_onnx[i]], color=colors[i], marker='o', label=model)
    axes[1, 0].plot([model], [a100_speed_trt[i]], color=colors[i], marker='o', label=model)
    axes[1, 1].plot([model], [params[i]], color=colors[i], marker='o', label=model)
    axes[1, 2].plot([model], [flops[i]], color=colors[i], marker='o', label=model)
    
    # 添加数据点的数值
    axes[0, 0].annotate(str(acc_top1[i]), (model, acc_top1[i]), xytext=xytext, textcoords='offset points')
    axes[0, 1].annotate(str(acc_top5[i]), (model, acc_top5[i]), xytext=xytext, textcoords='offset points')
    axes[0, 2].annotate(str(cpu_speed_onnx[i]), (model, cpu_speed_onnx[i]), xytext=xytext, textcoords='offset points')
    axes[1, 0].annotate(str(a100_speed_trt[i]), (model, a100_speed_trt[i]), xytext=xytext, textcoords='offset points')
    axes[1, 1].annotate(str(params[i]), (model, params[i]), xytext=xytext, textcoords='offset points')
    axes[1, 2].annotate(str(flops[i]), (model, flops[i]), xytext=xytext, textcoords='offset points')

# 添加标题
axes[0, 0].set_title('Accuracy @ Top-1 (%)')
axes[0, 1].set_title('Accuracy @ Top-5 (%)')
axes[0, 2].set_title('CPU Speed ONNX (ms)')
axes[1, 0].set_title('NVIDIA Tesla A100 Speed TensorRT (ms)')
axes[1, 1].set_title('Params (M)')
axes[1, 2].set_title('FLOPs (B)')    

# 添加大标题
plt.suptitle(r'The Comparison of YOLOv8-Classify @ $224\times 224$')

# 添加图例
axes[0, 0].legend()
axes[0, 1].legend()
axes[0, 2].legend()
axes[1, 0].legend()
axes[1, 1].legend()
axes[1, 2].legend()

# 调整子图间的间距
plt.tight_layout()

# 保存图片
save_path = os.path.join(save_dir, image_filename + filetype)
plt.savefig(save_path, dpi=dpi)
print(f"✅  The image has been save at {save_path}")