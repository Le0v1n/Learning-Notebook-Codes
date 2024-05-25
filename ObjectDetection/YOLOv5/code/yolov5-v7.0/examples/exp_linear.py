import sys
import os
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
from pylab import mpl
# 设置中文字体
mpl.rcParams["font.sans-serif"] = ["Times New Roman"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False


if __name__ == "__main__":
    y1 = 1
    y2 = 0.01
    epochs = 200
    
    func = lambda x: (1 - x / epochs) * (1.0 - y2) + y2  # linear
    
    x = [epoch for epoch in range(epochs)]
    lr = [func(epoch) for epoch in range(epochs)]
    
    # ---------- 画图 ----------
    plt.figure(dpi=200)
    plt.plot(x, lr)
    
    plt.title(f"Linear (epoch={epochs}, y1={y1}, y2={y2})")
    plt.xlabel("epoch", fontsize=14)
    plt.ylabel("lr", fontsize=14)
    
    plt.savefig("examples/lr-linear.png", dpi=200)
    