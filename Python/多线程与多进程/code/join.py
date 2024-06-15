import threading
import time


def worker(name, duration):
    print(f"线程 {name} 开始执行")
    time.sleep(duration)
    print(f"线程 {name} 完成")


def main():
    # 创建线程
    thread1 = threading.Thread(target=worker, args=(1, 2))
    thread2 = threading.Thread(target=worker, args=(2, 5))

    # 启动线程
    thread1.start()
    thread2.start()

    # 主线程等待线程1完成
    thread1.join()
    print("主线程等待线程1完成")

    # 此时线程2可能还在运行，但主线程可以继续执行其他任务
    print(f"主线程在线程2还在执行过程中打印了这行字！")

    # 主线程等待线程2完成
    thread2.join()
    print("主线程等待线程2完成")


if __name__ == "__main__":
    main()