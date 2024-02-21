# 6. YOLOv5 项目目录结构

```j
./                           # 📂YOLOv5项目的根目录
├── CITATION.cff                # (Citation File Format): 这是一个用于描述如何引用该软件项目的文件。它通常包含了软件的作者、版本号、发布年份、DOI（数字对象标识符）等信息。这有助于学术研究者在撰写论文时正确引用该软件，确保软件开发者的贡献得到认可。
├── CONTRIBUTING.md             # 这是一个指导文件，为潜在的贡献者提供了如何为项目贡献代码、文档或其他资源的指南。它可能包括项目的编码标准、提交准则、代码审查流程等。
├── LICENSE                     # 这是软件项目的许可证文件，规定了软件的使用、复制、修改和分发等权利和义务。开源项目的许可证通常遵循OSI（开放源代码倡议）认证的许可证，例如GPL、MIT、Apache等。
├── Le0v1n                      # 📂自己使用的测试代码
│   ├── plots-scheduler.py          # 绘制scheduler用的脚本
│   ├── results                     # 📂存放绘制结果的文件夹
│   ├── test-SPP.py                 # 测试SPP模块 
│   ├── test-SPP_SPPF-2.py          # 测试SPP模块
│   ├── test-SPP_SPPF.py            # 测试SPP模块
│   ├── test-focus-1.py             # 测试focus模块
│   └── test-focus-2.py             # 测试focus模块
├── README.md                   # 说明文件
├── README.zh-CN.md             # 说明文件（中文版）
├── __pycache__                 # 📂__pycache__目录和其中的.pyc文件是Python字节码的缓存。当Python源代码文件（.py）被解释器执行时，它会自动生成与源文件相对应的字节码文件（.pyc）。这些字节码文件可以被Python解释器更快地加载和执行，因为它们省去了每次运行时都需要将源代码转换为字节码的过程。
│   └── val.cpython-38.pyc          # 字节码缓存
├── benchmarks.py               # 给定模型（默认为YOLOv5s），该脚本会自动运行所有支持的格式（如onnx、openvino...），在coco128数据集上进行测试
├── classify                    # 📂将YOLOv5用于分类任务（Classification）
│   ├── predict.py                  # 预测脚本（images, videos, directories, globs, YouTube, webcam, streams, etc.）
│   ├── train.py                    # 训练基于YOLOv5的分类模型
│   ├── tutorial.ipynb              # 相关教程
│   └── val.py                      # 验证脚本
├── data                        # 📂存放不同数据集的配置文件
│   ├── Argoverse.yaml              # 一个用于自动驾驶的大规模、高多样性的数据集，包含了高清地图、传感器数据和交通代理的标注。它旨在支持自动驾驶系统的研究和开发，特别是那些依赖于高度详细的地图数据和精确的动态环境理解的系统。
│   ├── GlobalWheat2020.yaml        # 一个用于小麦叶锈病检测的数据集。它包含了大量的图像，旨在支持机器学习模型的发展，以便自动检测和识别这种作物病害
│   ├── ImageNet.yaml               # 一个大型的视觉数据库，用于视觉对象识别软件研究。它包含数百万个标注过的图像，涵盖了成千上万的类别。ImageNet挑战赛促进了深度学习在图像识别领域的快速发展
│   ├── ImageNet10.yaml             # ImageNet的子集，包含了20张图像（train和val各10张）。通常用于教育和研究目的，以便于在有限的资源和时间内进行实验。
│   ├── ImageNet100.yaml            # ImageNet的子集，包含了200张图像（train和val各100张）。通常用于教育和研究目的，以便于在有限的资源和时间内进行实验。
│   ├── ImageNet1000.yaml           # ImageNet的子集，包含了2000张图像（train和val各1000张）。通常用于教育和研究目的，以便于在有限的资源和时间内进行实验。
│   ├── Objects365.yaml             # 一个大规模的对象检测数据集，包含了365个类别的物体。它旨在推动计算机视觉领域的研究，特别是在对象检测和识别方面
│   ├── SKU-110K.yaml               # 一个大规模的商品识别数据集，包含了超过110,000个SKU（库存单位）的图像。它用于训练和评估机器学习模型，以便在零售环境中自动识别商品
│   ├── VOC.yaml                    # 一组用于视觉对象分类和检测的图像。它由PASCAL网络组织创建，并用于PASCAL VOC挑战赛，这是一个年度的计算机视觉竞赛
│   ├── VisDrone.yaml               # 一个大规模的无人机视角图像和视频数据集，用于视觉对象检测和跟踪。它涵盖了多种场景和对象类别，旨在支持无人机在智能监控和交通监控等领域的应用
│   ├── coco.yaml                   # 一个大型的图像数据集，用于对象检测、分割和字幕生成。它包含了超过30万张图像，涵盖了80个类别，并提供了精细的分割掩码和图像描述
│   ├── coco128-seg.yaml            # COCO128-seg是COCO数据集的子集，包含了80个类别的128张图像和相应的分割标注。通常用于原型设计和benchmark的测试。
│   ├── coco128.yaml                # COCO128是COCO数据集的子集，包含了80个类别的128张图像。通常用于原型设计和benchmark的测试。
│   ├── hyps                        # 📂存放超参数配置文件
│   │   ├── hyp.Objects365.yaml         # 用于Objects365数据集的超参数配置
│   │   ├── hyp.VOC.yaml                # 用于VOC数据集的超参数配置
│   │   ├── hyp.no-augmentation.yaml    # 不使用任何数据增强的超参数配置
│   │   ├── hyp.scratch-high.yaml       # 用于COCO数据集的“从头开始训练的”的超参数配置（拥有比较强的数据增强效果）
│   │   ├── hyp.scratch-low.yaml        # 用于COCO数据集的“从头开始训练的”的超参数配置（拥有比较弱的数据增强效果）
│   │   └── hyp.scratch-med.yaml        # 用于COCO数据集的“从头开始训练的”的超参数配置（拥有中间水平的数据增强效果）
│   ├── images                      # 📂存放用于测试的图片
│   │   ├── bus.jpg                    # 测试图片1
│   │   └── zidane.jpg                 # 测试图片2：“zidane.jpg” 是一张著名的图片，它展示了法国足球运动员齐内丁·齐达内（Zinedine Zidane）在2006年世界杯决赛中头顶意大利后卫马尔坎达利（Marco Materazzi）的场景。这张图片因其捕捉到了一个极具争议和情感高涨的体育时刻而闻名。
│   ├── scripts                     # 📂存放一些下载数据集、模型权重的shell脚本文件
│   │   ├── download_weights.sh         # 下载YOLOv5预训练权重的shell脚本
│   │   ├── get_coco.sh                 # 下载coco数据集（全量）的shell脚本
│   │   ├── get_coco128.sh              # 下载coco128数据集（coco128+coco128-seg）的shell脚本
│   │   ├── get_imagenet.sh             # 下载imagenet数据集（全量）的shell脚本
│   │   ├── get_imagenet10.sh           # 下载imagenet10数据集（20张图片的子集）的shell脚本
│   │   ├── get_imagenet100.sh          # 下载imagenet100数据集（200张图片的子集）的shell脚本
│   │   └── get_imagenet1000.sh         # 下载imagenet1000数据集（200张图片的子集）的shell脚本
│   └── xView.yaml                  # 一个用于目标检测的大规模遥感图像数据集，主要用于推动在空间图像上的计算机视觉研究和应用。这个数据集专注于针对自然灾害和人为事件的图像进行目标检测，如洪水、火灾、风暴等
├── detect.py                   # YOLOv5检测任务的预测脚本（images, videos, directories, globs, YouTube, webcam, streams, etc.）
├── dir_tree.txt                # 存放该项目下所有文件的说明文件（metadata）
├── export.py                   # 将YOLOv5模型导出为其他格式的模型（包含分类模型、检测模型、分割模型），支持众多格式：PyTorch、TorchScript、ONNX、OpenVINO、TensorRT、CoreML、TensorFlow SavedModel、TensorFlow GraphDef、TensorFlow Lite、TensorFlow Edge TPU、TensorFlow.js、PaddlePaddle
├── export2onnx.sh              # 自己编写的一个shell脚本，目的是方便复用
├── hubconf.py                  # 下载ultralytics提供的YOLOv5模型用的脚本（可以返回一个model变量供我们使用）
├── models                      # 📂存放YOLOv5的模型文件
│   ├── __init__.py                 # 用于将目录标识为包含Python模块的包
│   ├── common.py                   # 模型共用的模块存放文件，包括：SPPF、Conv、focus等等
│   ├── experimental.py             # 一些实验性的模块和函数
│   ├── hub                         # 📂存放YOLOv5的【目标检测】模型定义文件
│   │   ├── anchors.yaml                # 存放一些默认的Anchors尺寸模板
│   │   ├── yolov3-spp.yaml             # 使用SPP和YOLOv3模型定义文件
│   │   ├── yolov3-tiny.yaml            # YOLOv3-tiny的模型定义文件
│   │   ├── yolov3.yaml                 # YOLOv3的模型定义文件
│   │   ├── yolov5-bifpn.yaml           # 使用bi-FPN的YOLOv5模型定义文件
│   │   ├── yolov5-fpn.yaml             # 使用FPN的YOLOv5模型定义文件
│   │   ├── yolov5-p2.yaml              # 添加p2检测头的YOLOv5模型定义文件（4个检测头，默认为YOLOv5l）--> 小目标
│   │   ├── yolov5-p34.yaml             # 只使用p3和p4检测头的YOLOv5模型定义文件（默认使用的是p3、p4、p5）（2个检测头，默认为YOLOv5l）
│   │   ├── yolov5-p6.yaml              # 添加p6检测头的YOLOv5模型定义文件（4个检测头，默认为YOLOv5l）--> 大目标
│   │   ├── yolov5-p7.yaml              # 添加p6和p7检测头的YOLOv5模型定义文件（5个检测头，默认为YOLOv5l）--> 大大目标
│   │   ├── yolov5-panet.yaml           # 添加PaNet结构的模型定义文件
│   │   ├── yolov5l6.yaml               # 添加p6检测头的YOLOv5l模型定义文件（4个检测头）--> 大目标
│   │   ├── yolov5m6.yaml               # 添加p6检测头的YOLOv5m模型定义文件（4个检测头）--> 大目标
│   │   ├── yolov5n6.yaml               # 添加p6检测头的YOLOv5n模型定义文件（4个检测头）--> 大目标
│   │   ├── yolov5s-LeakyReLU.yaml      # 使用LeakyReLU的YOLOv5s模型定义文件
│   │   ├── yolov5s-ghost.yaml          # 使用Ghost模块替换普通卷积的的YOLOv5s模型定义文件
│   │   ├── yolov5s-transformer.yaml    # 使用Transform模块（C3TR）替换Backbone中最后一个C3模块的YOLOv5s模型定义文件
│   │   ├── yolov5s6.yaml               # 添加p6检测头的YOLOv5s模型定义文件（4个检测头）--> 大目标
│   │   └── yolov5x6.yaml               # 添加p6检测头的YOLOv5x模型定义文件（4个检测头）--> 大目标
│   ├── segment                     # 📂存放YOLOv5的【语义分割】模型定义文件
│   │   ├── yolov5l-seg.yaml            # 基于YOLOv5l的分割模型
│   │   ├── yolov5m-seg.yaml            # 基于YOLOv5m的分割模型
│   │   ├── yolov5n-seg.yaml            # 基于YOLOv5n的分割模型
│   │   ├── yolov5s-seg.yaml            # 基于YOLOv5s的分割模型
│   │   └── yolov5x-seg.yaml            # 基于YOLOv5x的分割模型
│   ├── tf.py                       # TensorFlow、Keras、TFLite版本的YOLOv5
│   ├── yolo.py                     # return model的脚本（包含了Classification、Det、Seg）
│   ├── yolo.sh                     # 对应的shell脚本，方便复用
│   ├── yolov5l.yaml                # YOLOv5l的目标检测模型定义
│   ├── yolov5m.yaml                # YOLOv5m的目标检测模型定义
│   ├── yolov5n.yaml                # YOLOv5n的目标检测模型定义
│   ├── yolov5s.yaml                # YOLOv5s的目标检测模型定义
│   └── yolov5x.yaml                # YOLOv5x的目标检测模型定义
├── pyproject.toml              # Python 项目的核心配置文件，它用于定义项目的元数据、依赖关系、构建系统和其它相关的配置信息。这个文件遵循 TOML（Tom’s Obvious, Minimal Language）格式，这是一种旨在作为小型的配置文件的人性化数据序列化格式。
├── requirements.txt            # 运行YOLOv5项目所需的第三方依赖库，可以通过 pip install -r requirements.txt 进行自动安装
├── runs                        # 📂YOLOv5运行产生的结果
│   └── train                       # 📂训练产生的结果的存放文件夹
│       ├── exp                         # 📂实验名称
│           ├── events.out.tfevents.1706866890.DESKTOP-PTPE509.23412.0  # TensorBoard的日志文件
│           ├── hyp.yaml                    # 模型训练使用的超参数
│           ├── labels.jpg                  # 训练集中所有标签（类别）的分布
│           ├── labels_correlogram.jpg      # 展示不同标签之间的相关性
│           ├── opt.yaml                    # 模型训练使用的配置
│           ├── train_batch0.jpg            # 训练过程中的几个批次（batch）的可视化结果
│           ├── train_batch1.jpg            # 训练过程中的几个批次（batch）的可视化结果
│           ├── train_batch2.jpg            # 训练过程中的几个批次（batch）的可视化结果
│           └── weights                     # 存放模型权重
├── segment                     # 📂分割任务使用的脚本
│   ├── predict.py                  # 分割任务的预测脚本
│   ├── train.py                    # 分割任务的训练脚本
│   ├── tutorial.ipynb              # 分割任务的教程
│   └── val.py                      # 分割任务的验证脚本
├── train.py                    # 目标检测任务的训练脚本
├── train.sh                    # 目标检测任务的训练脚本的shell文件，便于复用
├── tutorial.ipynb              # 目标检测任务的教程
├── utils                       # 📂常用工具（提高代码复用率）
│   ├── __init__.py
│   ├── __pycache__
│   ├── activations.py              # 存放常见的激活函数
│   ├── augmentations.py            # 存放常见的数据增强方法
│   ├── autoanchor.py               # 自动计算anchor大小的脚本
│   ├── autobatch.py                # 自动计算batch大小的脚本
│   ├── aws                         # 📂便于亚马逊aws服务的工具
│   │   ├── __init__.py
│   │   ├── mime.sh
│   │   ├── resume.py
│   │   └── userdata.sh
│   ├── callbacks.py                # 存放常用的回调函数
│   ├── dataloaders.py              # 存放常见的数据加载器
│   ├── docker                      # 📂用于构建Docker镜像的指令集
│   │   ├── Dockerfile                  # 定义如何构建应用程序的默认Docker镜像
│   │   ├── Dockerfile-arm64            # 类似于Dockerfile，但是它是专门为arm64架构（也称为aarch64）构建的
│   │   └── Dockerfile-cpu              # 仅使用CPU资源的场景构建的Docker镜像
│   ├── downloads.py                # 常用的下载工具
│   ├── flask_rest_api              # 📂轻量级的Web应用框架所用的api
│   │   ├── README.md                   # 说明文件
│   │   ├── example_request.py          # request的示例代码
│   │   └── restapi.py                  # Flask应用程序的主要入口点，其中定义了API的路由、视图函数以及可能的数据模型
│   ├── general.py                  # 更加通用的工具集合
│   ├── google_app_engine           # 📂Google App Engine相关文件
│   │   ├── Dockerfile                  # 定义如何构建Docker镜像
│   │   ├── additional_requirements.txt # 列出了项目所需的额外Python库
│   │   └── app.yaml                    # Google App Engine的配置文件，它告诉App Engine如何运行你的应用程序
│   ├── loggers                     # 📂存放日志相关文件
│   │   ├── __init__.py
│   │   ├── clearml                     # 📂用于机器学习实验跟踪、管理和自动化的平台
│   │   │   ├── README.md
│   │   │   ├── __init__.py
│   │   │   ├── clearml_utils.py
│   │   │   └── hpo.py
│   │   ├── comet                       # 📂用于机器学习实验跟踪的平台
│   │   │   ├── README.md
│   │   │   ├── __init__.py
│   │   │   ├── comet_utils.py
│   │   │   ├── hpo.py
│   │   │   └── optimizer_config.json
│   │   └── wandb                       # 📂用于机器学习实验跟踪的平台
│   │       ├── __init__.py
│   │       └── wandb_utils.py
│   ├── loss.py                     # 常用的损失函数
│   ├── metrics.py                  # 常用的指标评测方法
│   ├── plots.py                    # 常用的画图方法
│   ├── segment                     # 📂与分割任务相关的工具
│   │   ├── __init__.py
│   │   ├── augmentations.py            # 分割任务的数据增强方式
│   │   ├── dataloaders.py              # 分割任务的数据加载器
│   │   ├── general.py                  # 分割任务的更加通用的工具
│   │   ├── loss.py                     # 分割任务的损失函数
│   │   ├── metrics.py                  # 分割任务的评测指标
│   │   └── plots.py                    # 分割任务的画图方法
│   ├── torch_utils.py                  # 与PyTorch相关的工具
│   └── triton.py                       # NVIDIA的开源推理服务平台相关工具
├── val.py                      # 目标检测任务的验证脚本
└── weights                     # 📂存放预训练权重的文件夹
    ├── yolov5s-sim.onnx            # yolov5s的simplify版本的onnx模型
    ├── yolov5s.onnx                # yolov5s的onnx模型
    └── yolov5s.pt                  # yolov5s的pt模型

39 directories, 190 files
```














# 参考

1. 〔视频教程〕[YOLOv5入门到精通！不愧是公认的讲的最好的【目标检测全套教程】同济大佬12小时带我们从入门到进阶（YOLO/目标检测/环境部署+项目实战/Python/）](https://www.bilibili.com/video/BV1YG411876u?p=14)