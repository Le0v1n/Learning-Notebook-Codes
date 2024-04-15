import numpy as np
import matplotlib.pyplot as plt
# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'


# 定义激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def silu(x):
    return x * sigmoid(x)

def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

# 创建数据点
x = np.linspace(-10, 10, 400)
y_sigmoid = sigmoid(x)
y_silu = silu(x)
y_mish = mish(x)

# 绘制图形
plt.figure(figsize=(10, 6))

# 绘制激活函数
plt.plot(x, y_sigmoid, color='black', linestyle='--', label=r"$\rm{Sigmoid}=\frac{1}{1+e^{-x}}$")
plt.plot(x, y_silu, color="lightpink", label=r"$\rm{SiLU}=x \cdot \rm{Sigmoid}(x)$")
plt.plot(x, y_mish, color="skyblue", label=r"$\rm{Mish}=x \cdot \tanh(\ln(1+e^x))$")

# 不显示边框
plt.box(False)

# 绘制笛卡尔坐标系
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.ylim(-0.5, 1.5)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Different Activation Functions')
plt.legend()
plt.savefig("Le0v1n/results/activation_fn.jpg", dpi=200)
