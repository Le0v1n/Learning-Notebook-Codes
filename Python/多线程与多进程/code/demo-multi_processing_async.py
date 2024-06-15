import time
import random
from multiprocessing import Pool

def task_function(data):
    print(f"进程正在处理[数据-{data}]...")
    time.sleep(random.randint(3, 6))  # 随机等待3到6秒
    print(f"[数据-{data}]已被处理完毕！")
    return data

if __name__ == "__main__":
    with Pool(5) as pool:
        print("主进程定义了5个进程池中的工作进程")
        results = [pool.apply_async(task_function, args=(i,)) for i in range(5)]
        print("5个任务均已异步提交到进程池")

        # 处理结果
        for i, result in enumerate(results):
            try:
                print(f"任务-{i}的结果是：{result.get()}")
            except Exception as e:
                print(f"任务-{i}发生异常：{e}")

    pool.close()  # 关闭进程池，不再接受新任务
    pool.join()   # 等待所有子进程完成

    print("所有进程均执行完毕，主进程可以继续往下走了！")