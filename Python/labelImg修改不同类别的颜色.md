# 1. 问题

在我的任务中，有 `blue_obj`、 `green_obj` 和 `yellow_obj` ，但它们显示在不同颜色的框中，这使得我很难检查注释。我直接修改 box 的颜色后，下一张图片就不生效了。我应该如何将 box 的颜色修改为特定颜色？

例如：

+ <font color='orange'>blue_obj</font>：命名是蓝色，但 box 是黄色
+ <font color='blue'>red_obj</font>：命名是红色，但 box 是蓝色
+ <font color='red'>yellow_obj</font>：命名是黄色，但 box 是红色

以上这些框的颜色就会让人混乱。

# 2. 解决方案

为了解决这个问题，我们可以参考《[LabelImg目标检测标注工具之标记框修改显示明显、特定标签指定颜色](https://blog.csdn.net/qq_41767970/article/details/121958882)》这篇文章，具体做法为：

<kbd>Step 1</kbd> 下载 labelImg 的源码，下载地址为：[labelImg](https://github.com/HumanSignal/labelImg)

<kbd>Step 2</kbd> 根据 `Readme -> Installation -> Build from source`，按照不同操作系统安装相关依赖

```bash
pip install pyqt=5
pip install -c anaconda lxml
```
<kbd>Step 3</kbd> 修改 `labelImg\libs\utils.py` 文件中的函数：

```python
"""
    函数解释：根据text（即标签名）生成对应颜色
    修改自己需要的标签
"""
def generate_color_by_text(text):
    s = ustr(text)
    hash_code = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)
    r = int((hash_code / 255) % 255)
    g = int((hash_code / 65025) % 255)
    b = int((hash_code / 16581375) % 255)
    Q = QColor(r, g, b, 155)
    
    # RGBA格式: (R, G, B, 透明度)，范围均为 [0, 255]，其中透明度的越大越不透明，越小越透明
    if text == "类别1": # 类别1 设置为红色（完全不透明）
        Q = QColor(255, 0, 0, 255)
    elif text == "类别2": # 类别2 设置为绿色（完全不透明）
        Q = QColor(0, 255, 0, 255)
    elif text == "类别3": # 类别3 设置为蓝色（完全不透明）
        Q = QColor(0, 0, 255, 255)
    return Q
```

💡 **Tips**：其中 `类别` 指的是具体的类别名称，比如 `cat`、`dog` 这种。

<kbd>Step 4</kbd> 编译源码，使上述修改生效：

```bash
pyrcc5 -o libs/resources.py resources.qrc
```

<kbd>Step 5</kbd> 打开修改后的 `labelImg`：

```bash
python labelImg.py
```

# 3. 修改 box 的四个角的大小和 label 名称的大小

修改 `libs\shape.py` 文件：

```python
class Shape(object):
    P_SQUARE, P_ROUND = range(2)

    MOVE_VERTEX, NEAR_VERTEX = range(2)

    # The following class variables influence the drawing
    # of _all_ shape objects.
    line_color = DEFAULT_LINE_COLOR
    fill_color = DEFAULT_FILL_COLOR
    select_line_color = DEFAULT_SELECT_LINE_COLOR
    select_fill_color = DEFAULT_SELECT_FILL_COLOR
    vertex_fill_color = DEFAULT_VERTEX_FILL_COLOR
    h_vertex_fill_color = DEFAULT_HVERTEX_FILL_COLOR
    point_type = P_ROUND
    point_size = 8  # 点的大小
    scale = 1.0
    label_font_size = 8  # 标签的字体大小
```

💡 **Tips**：要想让每一个框在旁边显示其类别，可以使用快捷键 <kbd>ctrl + shift + p</kbd>。

# 4. 修改 [显示类别] 的文字大小

在 `labelImg.py` 中，修改：

```python
def paint_canvas(self):
    assert not self.image.isNull(), "cannot paint null image"
    self.canvas.scale = 0.01 * self.zoom_widget.value()
    self.canvas.overlay_color = self.light_widget.color()
    self.canvas.label_font_size = int(0.02 * max(self.image.width(), self.image.height()))
    self.canvas.adjustSize()
    self.canvas.update()
```

将 `self.canvas.label_font_size = int(0.02 * max(self.image.width(), self.image.height()))` 中的 `0.02` 改成为 `0.01` 或其他你想要的大小。

# 5. [补充] RBGA 颜色大全

[RGB颜色大全（HEX、对照表、Matplotlib、plt、好看的颜色）](https://blog.csdn.net/weixin_44878336/article/details/135003274)
