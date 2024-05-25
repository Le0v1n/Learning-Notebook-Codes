# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Auto-batch utils."""

from copy import deepcopy

import numpy as np
import torch

from utils.general import LOGGER, colorstr
from utils.torch_utils import profile


def check_train_batch_size(model, imgsz=640, amp=True):
    # Check YOLOv5 training batch size
    with torch.cuda.amp.autocast(amp):
        return autobatch(deepcopy(model).train(), imgsz)  # compute optimal batch size


def autobatch(model, imgsz=640, fraction=0.8, batch_size=16):
    """根据CUDA内存的`fraction`比例估算YOLOv5模型的最佳批处理大小。

    Args:
        model (_type_): 待估算的YOLOv5模型。
        imgsz (int, 可选): 输入图片的尺寸。默认为640。
        fraction (float, 可选): 期望使用的CUDA内存比例。默认为0.8。
        batch_size (int, 可选): 初始批处理大小。默认为16。

    Returns:
        _type_: 推荐的最佳batch size。
        
    Usage:
        import torch
        from utils.autobatch import autobatch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False)
        print(autobatch(model))
    """
    # Check device
    prefix = colorstr("AutoBatch: ")  # 给字符串添加颜色
    LOGGER.info(f"{prefix}Computing optimal batch size for --imgsz {imgsz}")
    device = next(model.parameters()).device  # get model device
    if device.type == "cpu":
        LOGGER.info(f"{prefix}CUDA not detected, using default CPU batch-size {batch_size}")
        # 如果不使用CUDA，那么默认返回batch_size=16
        return batch_size
    # 检查是否启用CuDNN库的benchmark模式。
    # CuDNN是一个用于深度学习的GPU加速库，它可以根据硬件和输入数据的特征自动选择最佳算法来加速卷积和其他操作。
    # benchmark模式会在每次运行时自动寻找最佳算法，但它可能会导致不稳定的结果。默认情况下，benchmark模式是禁用的
    if torch.backends.cudnn.benchmark:
        LOGGER.info(f"{prefix} ⚠️ Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}")
        return batch_size

    # 检查CUDA显存大小
    gb = 1 << 30  # bytes 转换为 GiB (1024 ** 3)

    # 获取设备完整名称，例子：'CUDA:0'
    d = str(device).upper()  # 'CUDA:0'

    # 获取CUDA设备的属性，例子：_CudaDeviceProperties(name='GeForce RTX 2080 Ti', major=7, minor=5, total_memory=11019MB, multi_processor_count=68)
    properties = torch.cuda.get_device_properties(device)  # device properties

    # 将显卡的总显存从原来的MB转换为GB，例子：10.76116943359375
    t = properties.total_memory / gb  # GiB total
    
    # 将显卡已经被程序预定的显存从MB转换为GB（这些显存我们的程序已经预定了，但还没有使用，
    # 这里在Terminal使用nvidia-smi可以查看，我这里是1114MiB / 11019MiB）
    r = torch.cuda.memory_reserved(device) / gb  # GiB reserved
    
    # 将显卡已经使用的显存从MB转换为GB
    a = torch.cuda.memory_allocated(device) / gb  # GiB allocated
    
    # 计算程序可用的显存（GB）
    f = t - (r + a)  # GiB free
    
    # 打印日志，例子：AutoBatch: CUDA:0 (GeForce RTX 2080 Ti) 10.76G total, 0.12G reserved, 0.05G allocated, 10.59G free
    LOGGER.info(f"{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free")

    # 分析在不同Batchsize下的性能
    batch_sizes = [1, 2, 4, 8, 16]
    try:
        # 创建batch分别为1,2,4,8,16的tensor，例子：[[1, 3, 640, 640], [2, 3, 640, 640], [4, 3, 640, 640], [8, 3, 640, 640], [16, 3, 640, 640]]
        img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]  # torch.empty()用于创建一个未初始化的张量（tensor），并分配内存空间给这个张量，其元素的值是未定义的，取决于内存中的内容。
        results = profile(img, model, n=3, device=device)  # 💡 这里的3是测试次数
        """💡 说明：results是一个list，每一个元素还是一个list，表示一张图片下模型的性能，包括：
            1. 模块参数
            2. flops
            3. 预定显存（GB）
            4. 输入图片大小
            5. 输出形式，例子：'list'
            
            例子：
                [
                    [7235389, 16.6252544, 0.281018368, 2188.4263356526694, 33580.41628201802, (...), 'list'], 
                    [7235389, 33.2505088, 0.478150656, 46.56608899434408, 32.1197509765625, (...), 'list'], 
                    [7235389, 66.5010176, 0.947912704, 34.093618392944336, 30.013322830200195, (...), 'list'], 
                    [7235389, 133.0020352, 1.778384896, 34.401098887125656, 33.556461334228516, (...), 'list'], 
                    [7235389, 266.0040704, 3.443523584, 59.91744995117188, 43.35379600524902, (...), 'list']
                ]
        """
    except Exception as e:
        LOGGER.warning(f"{prefix}{e}")

    # 求解最佳批处理大小
    y = [x[2] for x in results if x]  # 提取出所有的预定显存，例子：[0.281018368, 0.478150656, 0.947912704, 1.778384896, 3.443523584]

    # ---------- 使用y=ax+b进行拟合，其中batch_sizes[: len(y)]为x，y为y ----------
    # 说明：np.polyfit(y1, y2, deg=1)的作用是拟合一个一阶多项式（直线）到一组数据点 (y1, y2)。它根据给定的两个一维数组 y1 和 y2，返回拟合直线的系数。
    # 举例：我们有一组数据点 (y1, y2) = [(1, 2), (2, 4), (3, 6), (4, 8), (5, 10)]。通过调用np.polyfit(y1, y2, 1)，
    # 我们进行一阶多项式（直线）拟合，返回拟合系数 [2, 0]，表示拟合的直线方程为 y2 = 2*y1 + 0。
    p = np.polyfit(batch_sizes[: len(y)], y, deg=1)  # first degree polynomial fit
    # 例子：array([    0.21097,    0.077769])，即 y = 0.21097 * x + 0.077769
    # ----------------------------------------------------------------------------

    # 可分配显存大小f（GB）和期望使用的显存比例从而计算batchsize大小，例子：39
    b = int((f * fraction - p[1]) / p[0])  # y轴截距 (最佳批处理大小)

    # 如果batchsize=[1, 2, 4, 8, 16]中有些OOM了
    if None in results:  # some sizes failed
        i = results.index(None)  # 找到第一个OOM的batchsize的索引
        if b >= batch_sizes[i]:  # y轴截距高于失败点（计算出来的batchsize>=OOM的batchsize）
            b = batch_sizes[max(i - 1, 0)]  # 选择前一个安全的batchsize
    # 如果batchsize超出了指定范围
    if b < 1 or b > 1024:  # b outside of safe range
        b = batch_size
        LOGGER.warning(f"{prefix}WARNING ⚠️ CUDA anomaly detected, recommend restart environment and retry command.")

    # 计算刚才拟合的一阶多项式的值
    fraction = (np.polyval(p, b) + r + a) / t  # 实际预测的比例，例子：0.7879232085521087
    # 打印信息，例子：AutoBatch: Using batch-size 39 for CUDA:0 8.48G/10.76G (79%) ✅
    LOGGER.info(f"{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅")
    return b
