import os


# 创建一个深层的、不存在的文件夹
dirpath = 'Python/code/📂folder6/📂aaa/📂bbb/📂ccc/'

try:
    os.makedirs(dirpath)
    print(f"✅  {dirpath} 创建完毕！")
except Exception as e:
    print(f"❌  {e}")