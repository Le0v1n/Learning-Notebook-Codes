import time
import random
from multiprocessing import Process


def process_task(data):
    print(f"进程正在处理[数据-{data}]...")
    time.sleep(random.randint(3, 6))
    print(f"[数据-{data}]已被处理完毕！")
    
    
if __name__ == "__main__":
    processes = []
    for i in range(5):
        p = Process(
            target=process_task,
            args=(i, )
        )
        processes.append(p)
        p.start()  # 启动进程
        
    # 让主进程等待所有进程执行完毕后再执行（这是同步执行）
    print(f"在主进程中申请某个进程需要等待")
    for p in processes:
        p.join()
    print(f"所有进程均执行完毕，主进程可以继续往下走了！")