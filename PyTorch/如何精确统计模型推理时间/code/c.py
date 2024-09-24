import time
import torchvision
import torch
from tqdm import tqdm as TQDM
import numpy as np


def calc_inference_time_method_1(model: torch.nn.Module, input: torch.Tensor, times: int=50) -> float:
    with torch.inference_mode():
        time_list: list = []

        for _ in TQDM(range(times), desc='Method 1'):
            begin = time.perf_counter()
            results = model(input)
            end = time.perf_counter()
            time_list.append(end - begin)
            
    return np.average(time_list[5: ])


def calc_inference_time_method_2(model: torch.nn.Module, input: torch.Tensor, times: int=50) -> float:
    device = input.device
    with torch.inference_mode():
        time_list: list = []

        for _ in TQDM(range(times), desc='Method 2'):
            torch.cuda.synchronize(device=device)
            begin = time.perf_counter()
            results = model(input)
            torch.cuda.synchronize(device=device)
            end = time.perf_counter()
            time_list.append(end - begin)

    return np.average(time_list[5: ])


def calc_inference_time_method_3(model: torch.nn.Module, input: torch.Tensor, times: int=50) -> float:
    with torch.inference_mode():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        time_list: list = []

        for _ in TQDM(range(times), desc='Method 3'):
            start_event.record()
            results = model(input)
            end_event.record()
            end_event.synchronize()
            time_list.append(start_event.elapsed_time(end_event) / 1000)

    return np.average(time_list[5: ])


if __name__ == "__main__":
    # Create a model
    mobilenetv3 = torchvision.models.mobilenet_v3_large(weights=None, num_classes=1000, width_mult=1.0).cuda()

    # Create a dummy input
    dummpy_input = torch.randn(size=[32, 3, 224, 224]).cuda()

    time_1 = calc_inference_time_method_1(model=mobilenetv3, input=dummpy_input, times=50)
    print(f"The inference time of method 1: {time_1:.4f}s")

    time_2 = calc_inference_time_method_2(model=mobilenetv3, input=dummpy_input, times=50)
    print(f"The inference time of method 2: {time_2:.4f}s")

    time_3 = calc_inference_time_method_3(model=mobilenetv3, input=dummpy_input, times=50)
    print(f"The inference time of method 3: {time_3:.4f}s")
