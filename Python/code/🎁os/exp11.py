import os


try:
    os.mkdir(f"Python/code/📂folder2")  # 已经存在的文件夹
    print(f"✅  文件夹已创建！")
except Exception as e:
    print(f"{e = }")

try:
    os.mkdir(f"Python/code/📂folder3")  # 不存在的文件夹
    print(f"✅  文件夹已创建！")
except Exception as e:
    print(f"{e = }")