
<kbd>Note</kbd>：本文将 Matplotlib 命名为 plt：

```python
import matplotlib as plt
```

# 1. 将 plt 的输出图像嵌入 Jupyter Notebook 中

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Windows/ Mac系统提示副本错误时可以用该命令解决

# 使用ipython的魔法方法，将绘制出的图像直接嵌在notebook单元格中
import matplotlib.pyplot as plt
%matplotlib inline

# 设置绘图大小
plt.style.use({'figure.figsize':(10, 8)})
```

# 2. plt 输出中文

```python
import matplotlib.pyplot as plt
from pylab import mpl
# 设置中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False
```

# 3. plt 英文字体改为 Times New Roman

```python
import matplotlib.pyplot as plt
from matplotlib import rcParams


config = {
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)
```


