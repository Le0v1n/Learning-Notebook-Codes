import time
from multiprocessing import Pool


def task_function(x):
    time.sleep(1)
    print(f"\t进程执行函数并返回结果")
    return x


if __name__ == '__main__':
    processing_num = 4

    # 🪐 pool.apply()：同步的
    with Pool(processing_num) as pool:
        print(f"==================== 为每个进程分配相应的任务（同步的） ====================")
        results = [pool.apply(task_function, args=(i, )) for i in range(10)]
        
        # 展示任务的同步结果对象
        for i, processing in enumerate(results):
            print(f"任务-{i}：{processing}")
        print('-' * 50)
    print()
    
    # 🪐 pool.apply_async()：异步的
    with Pool(processing_num) as pool:  # 创建一个包含n个进程的进程池
        print(f"==================== 为每个进程分配相应的任务（异步的） ====================")
        results = [pool.apply_async(task_function, args=(i,)) for i in range(10)]
        print(f"---------- 分配异步任务环节已结束 ----------")
        
        # 展示任务的异步结果对象（AsyncResult 对象）
        for i, processing in enumerate(results):
            print(f"任务-{i}：{processing}")
        print('-' * 50)
            
        # 使用 get() 方法来获取每个异步任务的结果
        # 注意：只有当我们想要获取异步任务的结果时，这个任务才真正被进程所执行
        for i, result in enumerate(results):
            print(f"任务-{i}的结果是：{result.get()}")
        