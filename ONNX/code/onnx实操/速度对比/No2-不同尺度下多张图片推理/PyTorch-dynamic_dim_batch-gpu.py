import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision import models
import torch
import torch.nn.functional as F
import pandas as pd
import time
from rich.progress import track
import argparse


def parse_list(s):
    try:
        return list(map(int, s.strip('[]').split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError('Invalid list format. Must be comma-separated integers.')


# ==================================== 参数 ==================================== 
parser = argparse.ArgumentParser()
parser.add_argument('--images_folder', type=str, default='Datasets/Web/images', help='图片文件夹路径')
parser.add_argument('--input-shape', type=parse_list, default=[1, 3, 640, 640], help='The shape of input')
parser.add_argument('--test-times', type=int, default=50, help='The number of test')
parser.add_argument('--desc', type=str, default='[PyTorch-GPU]', help='The description of this test')
parser.add_argument('--warm-up', type=str, default='no', help='model warm-up')
parser.add_argument('--device', type=str, default='cuda', help='图片文件夹路径')
parser.add_argument('--verbose', action='store_true', help='')
args = parser.parse_args()  # 解析命令行参数
# ==============================================================================

# 加载 ONNX 模型，创建推理会话
model = models.mobilenet_v3_small(pretrained=True)  # ImageNet 预训练权重
model = model.eval().to(args.device)

if args.warm_up == 'yes':
    dummy_input = np.random.randn(args.input_shape[0], args.input_shape[1], args.input_shape[2], args.input_shape[3]).astype(np.float32)
    torch_output = model(dummy_input)
    print(f"模型冷启动完毕! 其推理结果的shape为: {torch_output.shape}") if args.verbose else ...
    
# 定义预处理函数
img_transform = transforms.Compose([
    transforms.Resize(args.input_shape[-1]),
    transforms.CenterCrop(args.input_shape[-1]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # imagenet专用
        std=[0.229, 0.224, 0.225]),  # imagenet专用
])

cost_times = [] 
for test_idx in track(range(args.test_times), description=args.desc):
    t1 = time.time()
    
    # 读取图片
    images_list = [os.path.join(args.images_folder, img) for img in os.listdir(args.images_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

    tensor_list = []
    for img_path in images_list:
        img = Image.open(fp=img_path)  # 读取图片
        input_img = img_transform(img)  # 图片预处理
        tensor_list.append(input_img)

    image_tensor = torch.stack(tensor_list)

    # ONNX模型推理
    inputs = image_tensor.to(args.device)
    print(f"inputs.type: {type(inputs)}") if args.verbose else ...
    print(f"inputs.shape: {inputs.shape}") if args.verbose else ...

    # 模型推理图片
    torch_results = model(inputs)
    print(f"模型推理完毕! 此时结果的shape为：{torch_results.shape}") if args.verbose else ...

    # ==================================== 后处理 ==================================== 
    # 使用 softmax 函数将分数转换为概率
    torch_results_softmax = F.softmax(input=torch_results, dim=1)
    print(f"经过softmax后的输出的shape为：{torch_results_softmax.shape}") if args.verbose else ...

    # 取概率最大的前 n 个结果
    n = 3
    top_n = torch.topk(input=torch_results_softmax, k=n, dim=1)

    probs = top_n.values  # torch.Size([18, 3])
    indices = top_n.indices  # torch.Size([18, 3])

    df = pd.read_csv('Datasets/imagenet_classes_indices.csv')

    idx2labels = {}
    for _image_idx, row in df.iterrows():
        # idx2labels[row['ID']] = row['class']  # 英文标签
        idx2labels[row['ID']] = row['Chinese']  # 中文标签

    print(f"============================== 推理结果 ==============================") if args.verbose else ...
    for _image_idx, img_name in enumerate(images_list):
        print(f"Image[{_image_idx}]: {img_name}") if args.verbose else ...
        _probs = probs[_image_idx].detach().cpu().numpy()
        _indices = indices[_image_idx].detach().cpu().numpy()
        
        for i, (prob, idx) in enumerate(zip(_probs, _indices)):
            class_name = idx2labels[idx]
            text = f"\tNo.{i}: {class_name:<30} --> {prob:>.4f}"
            print(text) if args.verbose else ...

    t2 = time.time()
    # print(f"[{desc}]-[{test_idx}]花费的时间为: {t2 - t1}")
    cost_times.append(t2 - t1)
    
def get_file_size(file_path):
    # 获取文件大小（字节）
    file_size_bytes = os.path.getsize(file_path)
    
    # 将字节转换为MB
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    return file_size_mb

print(f"\tAverage infer time: {np.average(cost_times):.4f}s\n")
    #   f"\tFile size:          {get_file_size('ONNX/saves/resnet18-f37072fd.pth'):.4f}MB")
with open("ONNX/saves/Speed_record-ResNet18.txt", 'a') as f:
    f.write(f"{np.average(cost_times):.4f}s|\n")