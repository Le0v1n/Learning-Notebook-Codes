import numpy as np
import matplotlib.pyplot as plt

# 已知数据点
x_known = np.array([1, 2, 3, 4, 5])
y_known = np.array([3, 5, 7, 9, 11])

# 待插值的数据点
x_unknown = [0.0, 1.5, 3.0, 4.5, 6.0]

# 使用np.interp进行插值
y_unknown = np.interp(x_unknown, x_known, y_known)
print(f"{y_unknown = }")  # [3, 4, 7, 10, 11]
exit()

# 绘制图形
plt.figure(figsize=(10, 6), dpi=200)
plt.plot(x_known, y_known, 'o', label='Known points', color='green')  # 已知数据点
plt.plot(x_unknown, y_unknown, 'o', label='Unknown points', color='red')  # 插值结果
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'Example for $np.interp()$')
plt.legend()
plt.grid(True)
plt.savefig('temp.jpg')
