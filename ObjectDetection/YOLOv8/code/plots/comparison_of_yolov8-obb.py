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
image_filename = 'Comparison_of_YOLOv8-obb'
filetype = '.png'
figsize = (20, 8)
dpi = 200
first_offset = (5, -5)
offset = (-20, -20)
# ==================================================================

# ---------- DATA ----------
models = ['v8n-obb', 'v8s-obb', 'v8m-obb', 'v8l-obb', 'v8x-obb']
mAP = [78.0, 79.5, 80.5, 80.7, 81.36]
cpu_speed_onnx = [204.77 , 424.88 , 763.48 , 1278.42, 1759.10]
a100_speed_trt = [3.57, 4.07, 7.61, 11.83, 13.23]
params = [3.1, 11.4, 26.4, 44.5, 69.5]
flops = [23.3, 76.3, 208.6, 433.8, 676.7]

# 创建画布
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize, dpi=dpi)

# 设置颜色循环
colors = plt.cm.tab10.colors

# 绘制折线
axes[0, 0].plot(models, mAP, color='#B0C4DE')
axes[0, 1].plot(models, cpu_speed_onnx, color='#B0C4DE')
axes[0, 2].plot(models, a100_speed_trt, color='#B0C4DE')
axes[1, 0].plot(models, params, color='#B0C4DE')
axes[1, 1].plot(models, flops, color='#B0C4DE')

# 绘制点
for i, model in enumerate(models):
    if i == 0:
        xytext = first_offset
    else:
        xytext = offset
    
    axes[0, 0].plot([model], [mAP[i]], color=colors[i], marker='o', label=model)
    axes[0, 1].plot([model], [cpu_speed_onnx[i]], color=colors[i], marker='o', label=model)
    axes[0, 2].plot([model], [a100_speed_trt[i]], color=colors[i], marker='o', label=model)
    axes[1, 0].plot([model], [params[i]], color=colors[i], marker='o', label=model)
    axes[1, 1].plot([model], [flops[i]], color=colors[i], marker='o', label=model)
    
    # 添加数据点的数值
    axes[0, 0].annotate(str(mAP[i]), (model, mAP[i]), xytext=xytext, textcoords='offset points')
    axes[0, 1].annotate(str(cpu_speed_onnx[i]), (model, cpu_speed_onnx[i]), xytext=xytext, textcoords='offset points')
    axes[0, 2].annotate(str(a100_speed_trt[i]), (model, a100_speed_trt[i]), xytext=xytext, textcoords='offset points')
    axes[1, 0].annotate(str(params[i]), (model, params[i]), xytext=xytext, textcoords='offset points')
    axes[1, 1].annotate(str(flops[i]), (model, flops[i]), xytext=xytext, textcoords='offset points')

# 添加标题
axes[0, 0].set_title(r'$\mathrm{mAP}^{\mathrm{test} \ @50}$ (%)')
axes[0, 1].set_title('CPU Speed ONNX (ms)')
axes[0, 2].set_title('NVIDIA Tesla A100 Speed TensorRT (ms)')
axes[1, 0].set_title('Params (M)')
axes[1, 1].set_title('FLOPs (B)')    

# 添加大标题
plt.suptitle(r'The Comparison of YOLOv8-OBB @ $1024\times 1024$')

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