<center><b><font size=12>YOLOv5：将模型转换为 ONNX</font></b></center>

📚 这个指南解释了如何将一个已经训练好的 YOLOv5 🚀 模型从 PyTorch 导出为 ONNX 和 TorchScript 格式。

<kbd>Key Words</kbd>：YOLOv5、onnx、trt、rknn、onnxruntime、模型转换

# 1. 模型格式

YOLOv5 推理官方支持 11 种格式：

| 格式 | 导出 .py 文件 | 模型文件 |
| :- | :- | :- |
| <kbd>PyTorch</kbd> | `yolov5s.pt` | `yolov5s.pt` |
| TorchScript | `torchscript` | `yolov5s.torchscript` |
| <kbd>ONNX</kbd> | `onnx` | `yolov5s.onnx` |
| OpenVINO | `openvino` | `yolov5s_openvino_model/` |
| <kbd>TensorRT</kbd> | `engine` | `yolov5s.engine` |
| CoreML | `coreml` | `yolov5s.mlmodel` |
| TensorFlow SavedModel | `saved_model` | `yolov5s_saved_model/` |
| TensorFlow GraphDef | `pb` | `yolov5s.pb` |
| TensorFlow Lite | `tflite` | `yolov5s.tflite` |
| TensorFlow Edge TPU | `edgetpu` | `yolov5s_edgetpu.tflite` |
| TensorFlow.js | `tfjs` | `yolov5s_web_model/` |
| <kbd>PaddlePaddle</kbd> | `paddle` | `yolov5s_paddle_model/` |

💡 **Tips**：

1. 导出到 ONNX 或 OpenVINO 可以获得高达 3 倍的 CPU 加速。请查看 [CPU 性能基准](https://github.com/ultralytics/yolov5/pull/6613)。
2. 导出到 TensorRT 可以获得高达 5 倍的 GPU 加速。请查看 [GPU 性能基准](https://github.com/ultralytics/yolov5/pull/6963)。

# 2. 导出已训练的 YOLOv5 模型

这个命令将一个预训练的 YOLOv5s 模型导出为 TorchScript 和 ONNX 格式。`yolov5s.pt` 是“小”型模型，是可用的第二小型模型。其他选项包括 `yolov5n.pt`、`yolov5m.pt`、`yolov5l.pt` 和 `yolov5x.pt`，以及它们的 P6 对应项，例如 `yolov5s6.pt`，或者可以使用自定义的训练检查点，例如 `runs/exp/weights/best.pt`。

```bash
python export.py --weights yolov5s.pt --include onnx --opset 12
```

💡 **Tips**：
1. 添加 `--half` 以以半精度 FP16 导出模型，以获得更小的文件大小；
2. 导出的 3 个模型将保存在原始的 PyTorch 模型旁边；
3. `--opset`: 设定 ONNX 版本

<div align=center>
    <img src=./imgs_markdown/2023-10-27-09-58-40.png
    width=80%>
</div>

# 3. 导出模型的使用示例

## 3.1 detect

`detect.py` 用于在导出的模型上运行推断：

```bash
python detect.py --weights yolov5s.pt                 # PyTorch
                           yolov5s.torchscript        # TorchScript
                           yolov5s.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                           yolov5s_openvino_model     # OpenVINO
                           yolov5s.engine             # TensorRT
                           yolov5s.mlmodel            # CoreML (macOS only)
                           yolov5s_saved_model        # TensorFlow SavedModel
                           yolov5s.pb                 # TensorFlow GraphDef
                           yolov5s.tflite             # TensorFlow Lite
                           yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                           yolov5s_paddle_model       # PaddlePaddle
```

💡 **Tips**：可以查看 `detect.py` 中有哪些参数，写好参数，上面给出的是简化版，并不适用于实际项目。

## 3.2 val

`val.py` 用于在导出的模型上运行验证：

```bash
python val.py --weights yolov5s.pt                 # PyTorch
                        yolov5s.torchscript        # TorchScript
                        yolov5s.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                        yolov5s_openvino_model     # OpenVINO
                        yolov5s.engine             # TensorRT
                        yolov5s.mlmodel            # CoreML (macOS Only)
                        yolov5s_saved_model        # TensorFlow SavedModel
                        yolov5s.pb                 # TensorFlow GraphDef
                        yolov5s.tflite             # TensorFlow Lite
                        yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                        yolov5s_paddle_model       # PaddlePaddle
```

💡 **Tips**：可以查看 `detect.py` 中有哪些参数，写好参数，上面给出的是简化版，并不适用于实际项目。

# 知识来源

1. [Ultralytics YOLOv5 Docs](https://docs.ultralytics.com/yolov5/)
2. [TFLite, ONNX, CoreML, TensorRT Export 🚀](https://docs.ultralytics.com/yolov5/tutorials/model_export/)