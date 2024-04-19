import os


dirpath = 'Python/code/📂folder1'
filepath = 'Python/code/📂folder1/exp1.py'

print(f"---------- os.path.isfile() 接收的是📂文件夹 ----------")
print(f"{os.path.isfile(dirpath) = }")

print(f"\n---------- os.path.isfile() 接收的是文件 ----------")
print(f"{os.path.isfile(filepath) = }")

print(f"\n---------- os.path.isfile(path) 接收的是不存在的路径 ----------")
print(f"{os.path.isfile('Python/code/XXXXX') = }")