import os


print(f"---------- os.path.isdir(path) 接收的是文件路径 ----------")
filepath = 'Python/code/📂folder2/exp2.txt'
flag = os.path.isdir(filepath)
print(f"{flag = }")

print(f"\n---------- os.path.isdir(path) 接收的是📂文件夹路径 ----------")
filepath = 'Python/code/📂folder2/📂sub_folder'
flag = os.path.isdir(filepath)
print(f"{flag = }")

print(f"\n---------- os.path.isdir(path) 接收的是不存在的路径 ----------")
filepath = 'Python/code/📂folder2/XXXXXX'
flag = os.path.isdir(filepath)
print(f"{flag = }")