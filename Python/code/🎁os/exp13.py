import os


print(f"---------- 使用 os.makedirs(exist_ok=False) 创建已经存在的文件夹 ----------")
dirpath = "Python/code/📂folder3"
try:
    os.makedirs(dirpath)  # 默认 exist_ok=False
    print(f"✅  文件夹 {dirpath} 已创建！")
except Exception as e:
    print(f"❌  {e = }")


print(f"\n---------- 使用 os.makedirs(exist_ok=True) 创建已经存在的文件夹 ----------")
dirpath = "Python/code/📂folder3"
try:
    os.makedirs(dirpath, exist_ok=True)
    print(f"✅  文件夹 {dirpath} 已创建！")
except Exception as e:
    print(f"❌  {e = }")
    
    
print(f"\n---------- 使用 os.makedirs(exist_ok=False) 创建已经不存在的文件夹 ----------")
dirpath = "Python/code/📂folder4"
try:
    os.makedirs(dirpath)  # 默认 exist_ok=False
    print(f"✅  文件夹 {dirpath} 已创建！")
except Exception as e:
    print(f"❌  {e = }")
    
    
print(f"\n---------- 使用 os.makedirs(exist_ok=True) 创建已经不存在的文件夹 ----------")
dirpath = "Python/code/📂folder5"
try:
    os.makedirs(dirpath, exist_ok=True)
    print(f"✅  文件夹 {dirpath} 已创建！")
except Exception as e:
    print(f"❌  {e = }")