# 1. 前置知识

## 1.1 损失函数

1. **Classification Loss**：
    - 用于衡量模型对目标的分类准确性。
    - 计算方式通常使用交叉熵损失函数，该函数衡量模型的分类输出与实际类别之间的差异。
    - 对于 YOLOv5，每个目标都有一个对应的类别，分类损失量化了模型对每个目标类别的分类准确性。
2. **Localization Loss：定位损失（预测边界框与 GT 之间的误差）**
    - 用于衡量模型对目标位置的预测准确性。
    - YOLOv5 中采用的是均方差（Mean Squared Error，MSE）损失函数，衡量模型对目标边界框坐标的回归预测与实际边界框之间的差异。
    - 定位损失关注模型对目标位置的精确度，希望模型能够准确地定位目标的边界框。
3. **Confidence Loss：置信度损失（框的目标性 <=> Objectness of the box）**
    - 用于衡量模型对目标存在与否的预测准确性。
    - YOLOv5 中采用的是二元交叉熵损失函数，该函数衡量模型对目标存在概率的预测与实际目标存在的二元标签之间的差异。
    - 置信度损失考虑了模型对每个边界框的目标置信度以及是否包含目标的预测。该损失鼓励模型提高对包含目标的边界框的预测概率，同时减小对不包含目标的边界框的预测概率。

总的损失函数：

$$
\rm Loss = \alpha \times Classification Loss + \beta \times Localization Loss + \gamma \times Confidence Loss
$$

## 1.2 PyTorch2ONNX

Netron 对 `.pt` 格式的兼容性不好，直接打卡无法显示整个网络。因此我们可以使用 YOLOv5 中的 `models/export.py` 脚本将 `.pt` 权重转换为 `.onnx` 格式，再使用 Netron 打开就可以完整地查看 YOLOv5 的整体架构了。

```bash
python models/export.py --weights weights/yolov5s.pt --img 640 --batch 1
```

## 1.3 

