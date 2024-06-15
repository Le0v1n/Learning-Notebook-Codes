import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor


def fetch_data_from_db(table_index):
    # 模拟数据库查询操作
    print(f"线程 {threading.current_thread().name} 正在从表 {table_index} 查询数据")
    time.sleep(random.randint(3, 6))  # 随机等待3~6秒
    

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=5) as executor:  # 创建线程池
        print('-' * 50)
        print("所有线程均开始执行各自的任务...")
        print('-' * 50)
        
        # 将任务提交给线程池
        future_to_thread = {executor.submit(fetch_data_from_db, i) for i in range(5)}

        # 等待所有任务完成
        for future in future_to_thread:
            future.result()  # 获取任务结果，如果有异常会在这里抛出

    print('-' * 50)
    print("所有任务均已完成。")
    print('-' * 50)