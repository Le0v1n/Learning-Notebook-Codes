import time
import random
import threading


def fetch_data_from_db(table_index):
    # 模拟数据库查询操作
    print(f"线程 {threading.current_thread().name} 正在从表 {table_index} 查询数据")
    time.sleep(random.randint(3, 6))  # 随机等待3~6秒


if __name__ == "__main__":
    # 创建线程列表
    threads = []
    for i in range(5):  # 假设有5个表，每个线程处理一个表
        thread = threading.Thread(target=fetch_data_from_db, args=(i,))
        threads.append(thread)
        thread.start()
    print('-' * 50)
    print(f"所有线程均开始执行各自的任务...")
    print('-' * 50)
    
    for thread in threads:
        print(f"线程 {thread.name} 正在等待其他线程执行完毕...")
        thread.join()  # 等待所有线程完成，这是同步执行