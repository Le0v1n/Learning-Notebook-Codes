import os


print(f"---------- 文件夹里面有文件夹 ----------")
dirpath = 'Python/code/📂folder6/'
try:
    os.rmdir(dirpath)
    print(f"✅  已成功删除 {dirpath} 文件夹！")
except Exception as e:
    print(f"❌  {e}")
    
    
print(f"\n---------- 文件夹里面有文件 ----------")
dirpath = 'Python/code/📂folder7/'
try:
    os.rmdir(dirpath)
    print(f"✅  已成功删除 {dirpath} 文件夹！")
except Exception as e:
    print(f"❌  {e}")