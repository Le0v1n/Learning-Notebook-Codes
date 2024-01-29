import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from rich.progress import track


# ==================================== 参数 ==================================== 
parser = argparse.ArgumentParser()
parser.add_argument('--image_folder_path', type=str, default='Datasets/Web/images', help='图片路径')
parser.add_argument('--input-shape', type=int, nargs=2, default=[256, 256])
parser.add_argument('--verbose', action='store_true', help='')
args = parser.parse_args()  # 解析命令行参数

onnx_weights = 'ONNX/saves/model-dynamic_dims.onnx'
onnx_weights_sim = 'ONNX/saves/model-dynamic_dims-sim.onnx'
# ==============================================================================

# 定义模型
onnx_model = onnxruntime.InferenceSession(path_or_bytes=onnx_weights)
onnx_model_sim = onnxruntime.InferenceSession(path_or_bytes=onnx_weights_sim)
# pytorch_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()  # ⚠️ 一定要 .eval
pytorch_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 定义预处理函数
img_transform = transforms.Compose([
    transforms.Resize(args.input_shape[-1]),
    transforms.CenterCrop(args.input_shape[-1]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # imagenet专用
        std=[0.229, 0.224, 0.225]),  # imagenet专用
])

image_list = [os.path.join(args.image_folder_path, img) for img in os.listdir(args.image_folder_path) \
               if img.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

for img_idx, image_path in track(enumerate(image_list), description='Precision Comparison'):
    # 读取图片
    img = Image.open(fp=image_path)  # 读取图片
    input_img = img_transform(img)
    input_img = input_img.unsqueeze(0)
    print(f"inputs.type: {type(input_img)}") if args.verbose else ...
    print(f"inputs.shape: {input_img.shape}") if args.verbose else ...

    model_ls = ['pt', 'onnx', 'onnx-sim']
    for model_name in model_ls:
        if model_name != 'pt':
            if not isinstance(input_img, np.ndarray):
                input_img = input_img.numpy()
            model_input = {'input': input_img, }
            model_result = onnx_model.run(output_names=['output'], input_feed=model_input)[0]
        else:
            model_result = pytorch_model(input_img)
        
        if not isinstance(model_result, torch.Tensor):
            model_result = torch.from_numpy(model_result)
        
        model_result_softmax = F.softmax(input=model_result, dim=1)  # [1, 1000]

        # 取概率最大的前 n 个结果
        n = 3
        top_n = torch.topk(input=model_result_softmax, k=n, dim=1)

        probs = top_n.values.detach().numpy()[0]  # torch.Size([18, 3])
        indices = top_n.indices.detach().numpy()[0]  # torch.Size([18, 3])
        print(f"probs: {probs}") if args.verbose else ...
        print(f"indices: {indices}") if args.verbose else ...

        df = pd.read_csv('Datasets/imagenet_classes_indices.csv')

        idx2labels = {}
        for _, row in df.iterrows():
            idx2labels[row['ID']] = row['Chinese']  # 中文标签

        print(f"============================== 推理结果-{model_name} ==============================")  if args.verbose else ...
        
        _results = []
        for i, (prob, idx) in enumerate(zip(probs, indices)):
            class_name = idx2labels[idx]
            text = f"No.{i}: {class_name:<30} --> {prob:>.5f}"  if args.verbose else ...
            _results.append(prob)
            print(text)
        print(f"=====================================================================")  if args.verbose else ...

        with open("ONNX/saves/Precision-comparison.txt", 'a') as f:
            if model_name == 'pt':
                f.write(f"|[{img_idx+1}] {os.path.basename(image_path)}"
                        f"|{_results[0]:>.5f}</br>{_results[1]:>.5f}</br>{_results[2]:>.5f}")
            elif model_name == 'onnx':
                f.write(f"|{_results[0]:>.5f}</br>{_results[1]:>.5f}</br>{_results[2]:>.5f}")
            else:
                f.write(f"|{_results[0]:>.5f}</br>{_results[1]:>.5f}</br>{_results[2]:>.5f}|\n")
                