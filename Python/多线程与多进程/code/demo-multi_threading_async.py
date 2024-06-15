import random
import asyncio


async def fetch_data_from_db_async(name):
    # 模拟数据库查询操作
    print(f"开始异步数据获取: {name}...")
    await asyncio.sleep(random.randint(3, 6))  # 随机等待3~6秒
    print(f"异步数据获取完成: {name}!")


async def main():
    # 创建5个异步任务，每个任务使用不同的名称
    tasks = [asyncio.create_task(fetch_data_from_db_async(f"任务-{i+1}")) for i in range(5)]
    
    # 等待所有任务完成
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())