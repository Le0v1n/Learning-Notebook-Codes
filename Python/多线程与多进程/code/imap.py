import time
import random
from multiprocessing import Pool
from tqdm import tqdm


def task_function(x):
    print(f"函数接收的数值为：{x}")
    time.sleep(random.randint(3, 6))  # 模拟数据处理过程
    return x ** 2


if __name__ == "__main__":
    with Pool(4) as pool:
        pbar = tqdm(
            iterable=pool.imap(
                func=task_function,
                iterable=range(30),
                chunksize=1
            ),
            total=30
        )
        
        for res in pbar:
            ...
            
    pbar.close()    
        
        # # 使用imap并行执行任务
        # # 💡 pool.imap会返回一个 IMapIterator 对象，但此时并不会开始执行任务
        # results = pool.imap(
        #     func=task_function,
        #     iterable=range(10),
        #     chunksize=1
        # )
        
        # # 只有当我们开始迭代这个 IMapIterator 对象（例如，使用 for 循环遍历 results）时，imap 方法才会开始执行。
        # # 每次迭代请求下一个结果。
        # for idx, obj in enumerate(results):
        #     print(f"[{idx}] {obj}")