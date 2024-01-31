import os
import sys
import platform
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
    

# Profile
import torch
import torch.nn as nn
from models.common import Focus, Conv, Bottleneck
from utils.torch_utils import profile 

    
if __name__ == "__main__":
    m1 = Focus(3, 64, 3)  # YOLOv5 Focus layer
    m2 = nn.Sequential(Conv(3, 32, 3, 1), Conv(32, 64, 3, 2), Bottleneck(64, 64))  # YOLOv3 first 3 layers

    # profile both 10 times at batch-size 16
    results = profile(input=torch.randn(16, 3, 640, 640), ops=[m1, m2], n=10, device='cpu')