import os


flag = os.path.exists('Python/code')
print(f"✅  文件夹存在") if flag else print(f"❌  文件夹不存在！")