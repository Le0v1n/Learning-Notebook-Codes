import os


print(f"---------- os.path.dirname(path) 接收的是文件路径 ----------")
filepath = 'Python/code/📂folder2/exp2.txt'
dirpath = os.path.dirname(filepath)
print(f"{dirpath = }")

print(f"\n---------- os.path.dirname(path) 接收的是文件夹路径 ----------")
filepath = 'Python/code/📂folder2/📂sub_folder'
dirpath = os.path.dirname(filepath)
print(f"{dirpath = }")

