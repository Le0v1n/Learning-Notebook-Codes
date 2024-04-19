import os
import sys


dst_path = 'Python/code'
all_files_path = []

for dirpath, dirnames, filenames in os.walk(dst_path):
    # dirpath: 本次遍历的文件夹路径
    # dirnames: 本次遍历得到的子文件夹名称
    # filenames: 本次遍历得到的文件名称
    for filename in filenames:
        all_files_path.append(os.path.join(dirpath, filename))

print(f"{all_files_path = }")
print(f"{len(all_files_path) = }")