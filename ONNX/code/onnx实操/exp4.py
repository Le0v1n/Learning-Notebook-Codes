import os
import random
import numpy as np
from PIL import Image
import onnxruntime
from torchvision import transforms
import torch
import torch.nn.functional as F
import pandas as pd


# ==================================== 参数 ==================================== 
onnx_weights_path = 'ONNX/saves/resnet18_imagenet-with_dynamic_axis.onnx'
images_folder = 'Datasets/Web/images'
infer_mode= 'single_batch'  # 推理模式: single_batch, multi_batch
warm_up = False  # 模型是否热身
# ============================================================================== 

# 加载 ONNX 模型，创建推理会话
ort_session = onnxruntime.InferenceSession(path_or_bytes=onnx_weights_path)  # ort -> onnxruntime

if warm_up:
    dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
    ort_inputs = {'input': dummy_input}
    ort_output = ort_session.run(output_names=['output'], input_feed=ort_inputs)[0]  # 输出被[]包围了，所以需要取出来
    print(f"模型冷启动完毕! 其推理结果的shape为: {ort_output.shape}")
    
# 定义预处理函数
img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # imagenet专用
        std=[0.229, 0.224, 0.225]),  # imagenet专用
])

# 读取图片
images_list = [os.path.join(images_folder, img) for img in os.listdir(images_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
random.shuffle(images_list)  # 打乱图片顺序

tensor_list = []
for img_path in images_list:
    img = Image.open(fp=img_path)  # 读取图片
    input_img = img_transform(img)  # 图片预处理
    tensor_list.append(input_img)

image_tensor = torch.stack(tensor_list)

# ONNX模型推理
inputs = image_tensor.numpy()
print(f"inputs.type: {type(inputs)}")
print(f"inputs.shape: {inputs.shape}")

# 模型推理图片
ort_inputs = {'input': inputs, }
ort_results = ort_session.run(output_names=['output'], input_feed=ort_inputs)[0]  # 得到 1000 个类别的分数
print(f"模型推理完毕! 此时结果的shape为：{ort_results.shape}")

# ==================================== 后处理 ==================================== 
# 使用 softmax 函数将分数转换为概率
ort_results_softmax = F.softmax(input=torch.from_numpy(ort_results), dim=1)
print(f"经过softmax后的输出的shape为：{ort_results_softmax.shape}")

# 取概率最大的前 n 个结果
n = 3
top_n = torch.topk(input=ort_results_softmax, k=n, dim=1)

probs = top_n.values  # torch.Size([18, 3])
indices = top_n.indices  # torch.Size([18, 3])

df = pd.read_csv('Datasets/imagenet_classes_indices.csv')

idx2labels = {}
for _image_idx, row in df.iterrows():
    # idx2labels[row['ID']] = row['class']  # 英文标签
    idx2labels[row['ID']] = row['Chinese']  # 中文标签

print(f"============================== 推理结果 ==============================")
for _image_idx, img_name in enumerate(images_list):
    print(f"Image[{_image_idx}]: {img_name}")
    _probs = probs[_image_idx].numpy()
    _indices = indices[_image_idx].numpy()
    
    for i, (prob, idx) in enumerate(zip(_probs, _indices)):
        class_name = idx2labels[idx]
        text = f"\tNo.{i}: {class_name:<30} --> {prob:>.4f}"
        print(text)
