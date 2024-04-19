import os


print(f"---------- os.path.split(path) 接收的是文件路径 ----------")
filepath = 'Python/code/📂folder2/exp2.txt'
dirpath, filename = os.path.split(filepath)
print(f"{dirpath = }")
print(f"{filename = }")
print(f"{os.path.dirname(filepath) = }")
print(f"{os.path.basename(filepath) = }")

print(f"\n---------- os.path.split(path) 接收的是📂文件夹路径 ----------")
filepath = 'Python/code/📂folder2/📂sub_folder'
dirpath, filename = os.path.split(filepath)
print(f"{dirpath = }")
print(f"{filename = }")
print(f"{os.path.dirname(filepath) = }")
print(f"{os.path.basename(filepath) = }")