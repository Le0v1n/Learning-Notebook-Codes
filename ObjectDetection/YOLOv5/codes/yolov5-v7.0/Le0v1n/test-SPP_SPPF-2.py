import sys
sys.path.append('/mnt/f/Projects/本地代码/Learning-Notebook-Codes/ObjectDetection/YOLOv5/codes/yolov5-v7.0')
import torch
import time
from tqdm.rich import tqdm
from models.common import SPP, SPPF


spp = SPP(c1=1024, c2=1024)
sppf = SPPF(c1=1024, c2=1024)

input_tensor = torch.randn(size=[1, 1024, 20, 20])
times = 200

t1 = time.time()
progress_bar = tqdm(total=times, desc='SPP')
for _ in range(times):
    tmp = spp(input_tensor)
    progress_bar.update()
progress_bar.close()
t2 = time.time()

progress_bar = tqdm(total=times, desc='SPPF')
for _ in range(times):
    tmp = sppf(input_tensor)
    progress_bar.update()
progress_bar.close()
t3 = time.time()

print(f"SPP (average time): {(t2 - t1) / times:.4f}s")
print(f"SPPF (average time): {(t3 - t2) / times:.4f}s")
