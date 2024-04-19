import os


print(f"---------- os.path.splitext(path) 接收的是文件路径 ----------")
filepath = 'Python/code/📂folder2/exp2.txt'
prefix, extension = os.path.splitext(os.path.basename(filepath))
print(f"{prefix = }")
print(f"{extension = }")

print(f"\n---------- os.path.splitext(path) 接收的是📂文件夹路径 ----------")
filepath = 'Python/code/📂folder2/📂sub_folder'
prefix, extension = os.path.splitext(os.path.basename(filepath))
print(f"{prefix = }")
print(f"{extension = }")