import time
import torchvision
import torch


if __name__ == "__main__":
    # Create a model
    mobilenetv3 = torchvision.models.mobilenet_v3_large(weights=None, num_classes=1000).cuda()

    # Create a dummy input
    dummpy_input = torch.ones(size=[4, 3, 224, 224]).cuda()

    begin = time.perf_counter()

    result = mobilenetv3(dummpy_input)

    end = time.perf_counter()

    # Statistics
    print(f"The inference time: {end - begin:.4f}s")
