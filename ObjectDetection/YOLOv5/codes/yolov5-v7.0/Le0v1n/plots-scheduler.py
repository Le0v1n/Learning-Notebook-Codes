import matplotlib.pyplot as plt
import math


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


# 设定训练的总epoch数
epochs = 100

# YOLOv5中的超参数
hyp = {
    "lr0": 0.01,  # 初始学习率
    "lrf": 0.1  # final OneCycleLR learning rate (lr0 * lrf)
}

# 创建一个numpy数组，表示epoch数
epoch_lst = range(epochs)

# Cosine调度器的学习率变化
lf_cos = one_cycle(1, hyp["lrf"], epochs)
lr_cos = [lf_cos(epoch) for epoch in epoch_lst]

# Linear调度器的学习率变化
lf_lin = lambda x: (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]
lr_lin = [lf_lin(epoch) for epoch in epoch_lst]

# 绘制学习率变化曲线
plt.figure(figsize=(10, 6), dpi=200)

plt.plot(epoch_lst, lr_cos, '-', label='Cosine Scheduler', color='skyblue')
plt.plot(epoch_lst, lr_lin, '-.', label='Linear Scheduler', color='lightpink')

plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Comparison of Cosine and Linear Learning Rate Schedulers')

plt.legend()
plt.grid(True)
plt.savefig('Le0v1n/results/Comparison-of-Cosine-and-Linear-Learning-Rate-Schedulers.jpg')