import os
import random
import numpy as np
from PIL import Image
import onnxruntime
from torchvision import transforms
import torch
import torch.nn.functional as F
import pandas as pd


# ==================================== 加载 ONNX 模型，创建推理会话 ==================================== 
ort_session = onnxruntime.InferenceSession(path_or_bytes='ONNX/saves/resnet18_imagenet-fix_axis.onnx')  # ort -> onnxruntime

# ==================================== 模型冷启动 ==================================== 
dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
ort_inputs = {'input': dummy_input}
ort_output = ort_session.run(output_names=['output'], input_feed=ort_inputs)[0]  # 输出被[]包围了，所以需要取出来
print(f"模型冷启动完毕! 其推理结果的shape为: {ort_output.shape}")

# ==================================== 加载真正的图像 ==================================== 
images_folder = 'Datasets/Web/images'
images_list = [os.path.join(images_folder, img) for img in os.listdir(images_folder) if img.lower().endswith(('.jpg', '.png', '.webp'))]

img_path = images_list[random.randint(0, len(images_list)-1)]
img = Image.open(fp=img_path)

# ==================================== 图像预处理 ==================================== 
# 定义预处理函数
img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # imagenet专用
        std=[0.229, 0.224, 0.225]),  # imagenet专用
])

# 对图片进行预处理
input_img = img_transform(img)
print(f"input_img.type: {type(input_img)}")
print(f"input_img.shape: {input_img.shape}")

# 为图片添加batch维度
input_img = torch.unsqueeze(input_img, dim=0)

# ==================================== ONNX模型推理 ==================================== 
# 因为ONNXRuntime需要的是numpy而非torch的tensor, 所以将其转换为numpy
input_img = input_img.numpy()
print(f"input_img.type: {type(input_img)}")
print(f"input_img.shape: {input_img.shape}")

# 模型推理图片
ort_inputs = {'input': input_img, }
ort_results = ort_session.run(output_names=['output'], input_feed=ort_inputs)[0]  # 得到 1000 个类别的分数
print(f"模型推理完毕! 此时结果的shape为：{ort_results.shape}")

# ==================================== 后处理 ==================================== 
# 使用 softmax 函数将分数转换为概率
ort_results_softmax = F.softmax(input=torch.from_numpy(ort_results), dim=1)
print(f"经过softmax后的输出的shape为：{ort_results_softmax.shape}")

# 取概率最大的前 n 个结果
n = 3
top_n = torch.topk(input=ort_results_softmax, k=n, dim=1)

probs = top_n.values.numpy()[0]
indices = top_n.indices.numpy()[0]

print(f"置信度最高的前{n}个结果为：\t{probs}\n"
      f"对应的类别索引为：\t\t{indices}")

# ==================================== 显示类别 ==================================== 
df = pd.read_csv('Datasets/imagenet_classes_indices.csv')

idx2labels = {}
for idx, row in df.iterrows():
    # idx2labels[row['ID']] = row['class']  # 英文标签
    idx2labels[row['ID']] = row['Chinese']  # 中文标签

print(f"=============== 推理结果 ===============\n"
      f"图片路径: {img_path}")
for i, (class_prob, idx) in enumerate(zip(probs, indices)):
    class_name = idx2labels[idx]
    text = f"\tNo.{i}: {class_name:<30} --> {class_prob:>.4f}"
    print(text)
