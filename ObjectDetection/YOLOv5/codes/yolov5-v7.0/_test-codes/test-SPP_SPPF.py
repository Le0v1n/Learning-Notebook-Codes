import sys
sys.path.append('/mnt/f/Projects/本地代码/Learning-Notebook-Codes/ObjectDetection/YOLOv5/codes/yolov5-v7.0')
from torchsummary import summary
from models.common import SPP, SPPF


spp = SPP(c1=32, c2=3)
sppf = SPPF(c1=32, c2=3)

summary(spp, (32, 26, 26))
summary(sppf, (32, 26, 26))
