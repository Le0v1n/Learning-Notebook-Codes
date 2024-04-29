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
image_filename = 'Comparison_of_YOLOv8-seg'
filetype = '.png'
figsize = (20, 8)
dpi = 200
first_offset = (5, -5)
offset = (-20, -20)
# ==================================================================

# ---------- DATA ----------
models = ['v8n-Seg', 'v8s-Seg', 'v8m-Seg', 'v8l-Seg', 'v8x-Seg']
mAP_box = [36.7, 44.6, 49.9, 52.3, 53.4]
mAP_mask = [30.5, 36.8, 40.8, 42.6, 43.4]
cpu_speed_onnx = [96.1, 155.7, 317.0, 572.4, 712.1]
a100_speed_trt = [1.21, 1.47, 2.18, 2.79, 4.02]
params = [3.4, 11.8, 27.3, 46.0, 71.8]
flops = [12.6, 42.6, 110.2, 220.5, 344.1]

# 创建画布
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize, dpi=dpi)

# 设置颜色循环
colors = plt.cm.tab10.colors

# 绘制折线
axes[0, 0].plot(models, mAP_box, color='#B0C4DE')
axes[0, 1].plot(models, mAP_mask, color='#B0C4DE')
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
    
    axes[0, 0].plot([model], [mAP_box[i]], color=colors[i], marker='o', label=model)
    axes[0, 1].plot([model], [mAP_mask[i]], color=colors[i], marker='o', label=model)
    axes[0, 2].plot([model], [cpu_speed_onnx[i]], color=colors[i], marker='o', label=model)
    axes[1, 0].plot([model], [a100_speed_trt[i]], color=colors[i], marker='o', label=model)
    axes[1, 1].plot([model], [params[i]], color=colors[i], marker='o', label=model)
    axes[1, 2].plot([model], [flops[i]], color=colors[i], marker='o', label=model)
    
    # 添加数据点的数值
    axes[0, 0].annotate(str(mAP_box[i]), (model, mAP_box[i]), xytext=xytext, textcoords='offset points')
    axes[0, 1].annotate(str(mAP_mask[i]), (model, mAP_mask[i]), xytext=xytext, textcoords='offset points')
    axes[0, 2].annotate(str(cpu_speed_onnx[i]), (model, cpu_speed_onnx[i]), xytext=xytext, textcoords='offset points')
    axes[1, 0].annotate(str(a100_speed_trt[i]), (model, a100_speed_trt[i]), xytext=xytext, textcoords='offset points')
    axes[1, 1].annotate(str(params[i]), (model, params[i]), xytext=xytext, textcoords='offset points')
    axes[1, 2].annotate(str(flops[i]), (model, flops[i]), xytext=xytext, textcoords='offset points')

# 添加标题
axes[0, 0].set_title(r'$\mathrm{mAP}^{\mathrm{box} \ @50-95}$ (%)')
axes[0, 1].set_title(r'$\mathrm{mAP}^{\mathrm{mask} \ @50-95}$ (%)')
axes[0, 2].set_title('CPU Speed ONNX (ms)')
axes[1, 0].set_title('NVIDIA Tesla A100 Speed TensorRT (ms)')
axes[1, 1].set_title('Params (M)')
axes[1, 2].set_title('FLOPs (B)')    

# 添加大标题
plt.suptitle(r'The Comparison of YOLOv8-Seg @ $640\times 640$')

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