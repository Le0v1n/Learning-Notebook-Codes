import time
import torchvision
import torch
from tqdm import tqdm as TQDM
import numpy as np


def cpu_task() -> None:
    x = np.random.randn(1, 3, 512, 512)
    x = x.astype(np.float32)
    x = x * 1024 ** 0.5


def calc_inference_time_method_1(model_1: torch.nn.Module, model_2: torch.nn.Module, input: torch.Tensor, times: int=50) -> tuple:
    with torch.inference_mode():
        time_list_1: list = []
        time_list_2: list = []

        for _ in TQDM(range(times), desc='Method 1'):
            # step.1: model 1
            begin_1 = time.perf_counter()
            results_1 = model_1(input)
            end_1 = time.perf_counter()

            # step.2: CPU
            cpu_task()

            # step.3: model 2
            begin_2 = time.perf_counter()
            results_2 = model_2(input)
            end_2 = time.perf_counter()

            time_list_1.append(end_1 - begin_1)
            time_list_2.append(end_2 - begin_2)
            
    return np.average(time_list_1[5: ]), np.average(time_list_2[5: ])


def calc_inference_time_method_2(model_1: torch.nn.Module, model_2: torch.nn.Module, input: torch.Tensor, times: int=50) -> tuple:
    device = input.device
    with torch.inference_mode():
        time_list_1: list = []
        time_list_2: list = []

        for _ in TQDM(range(times), desc='Method 2'):
            # step.1: model 1
            torch.cuda.synchronize(device=device)
            begin_1 = time.perf_counter()
            results_1 = model_1(input)
            torch.cuda.synchronize(device=device)
            end_1 = time.perf_counter()

            # step.2: CPU
            cpu_task()

            # step.3: model 2
            torch.cuda.synchronize(device=device)
            begin_2 = time.perf_counter()
            results_2 = model_2(input)
            torch.cuda.synchronize(device=device)
            end_2 = time.perf_counter()

            time_list_1.append(end_1 - begin_1)
            time_list_2.append(end_2 - begin_2)

    return np.average(time_list_1[5: ]), np.average(time_list_2[5: ])


def calc_inference_time_method_3(model_1: torch.nn.Module, model_2: torch.nn.Module, input: torch.Tensor, times: int=50) -> tuple:
    with torch.inference_mode():
        start_event_1 = torch.cuda.Event(enable_timing=True)
        end_event_1 = torch.cuda.Event(enable_timing=True)
        time_list_1: list = []

        start_event_2 = torch.cuda.Event(enable_timing=True)
        end_event_2 = torch.cuda.Event(enable_timing=True)
        time_list_2: list = []

        for _ in TQDM(range(times), desc='Method 3'):
            # Step.1 model 1
            start_event_1.record()
            results_1 = model_1(input)
            end_event_1.record()
            end_event_1.synchronize()

            # Step.2 CPU
            cpu_task()

            # Step.3 model 2
            start_event_2.record()
            results_2 = model_1(input)
            end_event_2.record()
            end_event_2.synchronize()

            time_list_1.append(start_event_1.elapsed_time(end_event_1) / 1000)
            time_list_2.append(start_event_2.elapsed_time(end_event_2) / 1000)

    return np.average(time_list_1[5: ]), np.average(time_list_2[5: ])


if __name__ == "__main__":
    # Create a model
    mobilenetv3_large = torchvision.models.mobilenet_v3_large(weights=None, num_classes=1000, width_mult=1.0).cuda()
    mobilenetv3_small = torchvision.models.mobilenet_v3_small(weights=None, num_classes=1000, width_mult=1.0).cuda()

    # Create a dummy input
    dummpy_input = torch.randn(size=[32, 3, 224, 224]).cuda()

    method_1_time_1, method_1_time_2 = calc_inference_time_method_1(model_1=mobilenetv3_large, model_2=mobilenetv3_small, input=dummpy_input, times=50)
    print(
        f"[Method 1]\n"
        f"\tThe inference time of model 1: {method_1_time_1:.4f}s\n"
        f"\tThe inference time of model 2: {method_1_time_2:.4f}s\n"
        f"\tThe average time: {np.average([method_1_time_1, method_1_time_2]):.4f}s\n"
    )

    method_2_time_1, method_2_time_2 = calc_inference_time_method_2(model_1=mobilenetv3_large, model_2=mobilenetv3_small, input=dummpy_input, times=50)
    print(
        f"[Method 2]\n"
        f"\tThe inference time of model 1: {method_2_time_1:.4f}s\n"
        f"\tThe inference time of model 2: {method_2_time_2:.4f}s\n"
        f"\tThe average time: {np.average([method_2_time_1, method_1_time_2]):.4f}s\n"
    )

    method_3_time_1, method_3_time_2 = calc_inference_time_method_3(model_1=mobilenetv3_large, model_2=mobilenetv3_small, input=dummpy_input, times=50)
    print(
        f"[Method 2]\n"
        f"\tThe inference time of model 1: {method_3_time_1:.4f}s\n"
        f"\tThe inference time of model 2: {method_3_time_2:.4f}s\n"
        f"\tThe average time: {np.average([method_3_time_1, method_1_time_2]):.4f}s\n"
    )
