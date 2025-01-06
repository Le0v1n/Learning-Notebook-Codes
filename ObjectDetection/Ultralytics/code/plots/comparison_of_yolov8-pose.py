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
image_filename = 'Comparison_of_YOLOv8-pose'
filetype = '.png'
figsize = (20, 8)
dpi = 200
first_offset = (5, -5)
offset = (-20, -20)
# ==================================================================

# ---------- DATA ----------
models = ['v8n-pose', 'v8s-pose', 'v8m-pose', 'v8l-pose', 'v8x-pose', 'v8x-pose-p6']
mAP_pose_5095 = [50.4, 60.0, 65.0, 67.6, 69.2, 71.6]
mAP_pose_50 = [80.1, 86.2, 88.8, 90.0, 90.2, 91.2]
cpu_speed_onnx = [131.8, 233.2, 456.3 , 784.5 , 1607.1, 4088.7]
a100_speed_trt = [1.18, 1.42, 2.00, 2.59, 3.73, 10.04]
params = [3.3, 11.6, 26.4, 44.4, 69.4, 99.1]
flops = [9.2, 30.2, 81.0, 168.6, 263.2, 1066.4]

# 创建画布
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize, dpi=dpi)

# 设置颜色循环
colors = plt.cm.tab10.colors

# 绘制折线
axes[0, 0].plot(models, mAP_pose_5095, color='#B0C4DE')
axes[0, 1].plot(models, mAP_pose_50, color='#B0C4DE')
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
    
    axes[0, 0].plot([model], [mAP_pose_5095[i]], color=colors[i], marker='o', label=model)
    axes[0, 1].plot([model], [mAP_pose_50[i]], color=colors[i], marker='o', label=model)
    axes[0, 2].plot([model], [cpu_speed_onnx[i]], color=colors[i], marker='o', label=model)
    axes[1, 0].plot([model], [a100_speed_trt[i]], color=colors[i], marker='o', label=model)
    axes[1, 1].plot([model], [params[i]], color=colors[i], marker='o', label=model)
    axes[1, 2].plot([model], [flops[i]], color=colors[i], marker='o', label=model)
    
    # 添加数据点的数值
    axes[0, 0].annotate(str(mAP_pose_5095[i]), (model, mAP_pose_5095[i]), xytext=xytext, textcoords='offset points')
    axes[0, 1].annotate(str(mAP_pose_50[i]), (model, mAP_pose_50[i]), xytext=xytext, textcoords='offset points')
    axes[0, 2].annotate(str(cpu_speed_onnx[i]), (model, cpu_speed_onnx[i]), xytext=xytext, textcoords='offset points')
    axes[1, 0].annotate(str(a100_speed_trt[i]), (model, a100_speed_trt[i]), xytext=xytext, textcoords='offset points')
    axes[1, 1].annotate(str(params[i]), (model, params[i]), xytext=xytext, textcoords='offset points')
    axes[1, 2].annotate(str(flops[i]), (model, flops[i]), xytext=xytext, textcoords='offset points')

# 添加标题
axes[0, 0].set_title(r'$\mathrm{mAP}^{\mathrm{pose} \ @50-95}$ (%)')
axes[0, 1].set_title(r'$\mathrm{mAP}^{\mathrm{pose} \ @50}$ (%)')
axes[0, 2].set_title('CPU Speed ONNX (ms)')
axes[1, 0].set_title('NVIDIA Tesla A100 Speed TensorRT (ms)')
axes[1, 1].set_title('Params (M)')
axes[1, 2].set_title('FLOPs (B)')    

# 添加大标题
plt.suptitle(r'The Comparison of YOLOv8-Pose @ $640\times 640$')

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