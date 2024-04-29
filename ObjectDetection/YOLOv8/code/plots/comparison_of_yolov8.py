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
image_filename = 'Comparison_of_YOLOv8'
filetype = '.png'
figsize = (20, 8)
dpi = 200
first_offset = (5, -5)
offset = (-20, -20)
# ==================================================================

# ---------- DATA ----------
models = ['v8n', 'v8s', 'v8m', 'v8l', 'v8x']
mAP = [37.3, 44.9, 50.2, 52.9, 53.9]
cpu_speed = [80.4, 128.4, 234.7, 375.2, 479.1]
a100_speed = [0.99, 1.20, 1.83, 2.39, 3.53]
params = [3.2, 11.2, 25.9, 43.7, 68.2]
flops = [8.7, 28.6, 78.9, 165.2, 257.8]

# 创建画布
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize, dpi=dpi)

# 设置颜色循环
colors = plt.cm.tab10.colors

# 绘制折线
axes[0, 0].plot(models, mAP, color='#B0C4DE')
axes[0, 1].plot(models, cpu_speed, color='#B0C4DE')
axes[0, 2].plot(models, a100_speed, color='#B0C4DE')
axes[1, 0].plot(models, params, color='#B0C4DE')
axes[1, 1].plot(models, flops, color='#B0C4DE')

# 绘制点
for i, model in enumerate(models):
    if i == 0:
        xytext = first_offset
    else:
        xytext = offset
    
    axes[0, 0].plot([model], [mAP[i]], color=colors[i], marker='o', label=model)
    axes[0, 1].plot([model], [cpu_speed[i]], color=colors[i], marker='o', label=model)
    axes[0, 2].plot([model], [a100_speed[i]], color=colors[i], marker='o', label=model)
    axes[1, 0].plot([model], [params[i]], color=colors[i], marker='o', label=model)
    axes[1, 1].plot([model], [flops[i]], color=colors[i], marker='o', label=model)
    
    # 添加数据点的数值
    axes[0, 0].annotate(str(mAP[i]), (model, mAP[i]), xytext=xytext, textcoords='offset points')
    axes[0, 1].annotate(str(cpu_speed[i]), (model, cpu_speed[i]), xytext=xytext, textcoords='offset points')
    axes[0, 2].annotate(str(a100_speed[i]), (model, a100_speed[i]), xytext=xytext, textcoords='offset points')
    axes[1, 0].annotate(str(params[i]), (model, params[i]), xytext=xytext, textcoords='offset points')
    axes[1, 1].annotate(str(flops[i]), (model, flops[i]), xytext=xytext, textcoords='offset points')

# 添加标题
axes[0, 0].set_title(r'$\mathrm{mAP}^{@50-95}$ (%)')
axes[0, 1].set_title('CPU Speed (ms)')
axes[0, 2].set_title('NVIDIA Tesla A100 Speed (ms)')
axes[1, 0].set_title('Params (M)')
axes[1, 1].set_title('FLOPs (B)')    

# 添加大标题
plt.suptitle(r'The Comparison of YOLOv8 @ $640\times 640$')

# 添加图例
axes[0, 0].legend()
axes[0, 1].legend()
axes[0, 2].legend()
axes[1, 0].legend()
axes[1, 1].legend()

# 调整子图间的间距
plt.tight_layout()

# 保存图片
save_path = os.path.join(save_dir, image_filename + filetype)
plt.savefig(save_path, dpi=dpi)
print(f"✅  The image has been save at {save_path}")