import os


def is_exists(path):
    if not os.path.exists(parent_dir):
        print(f"⚠️  {path} 不存在!")
        return False
    else:
        return True


parent_dir = 'Python/docs'
dirname = '📂folder1'

is_exists(parent_dir)

dirpath = os.path.join(parent_dir, dirname)
is_exists(dirpath)

try:
    os.mkdir(dirpath)  # 已经存在的文件夹
    print(f"✅  {dirpath} 文件夹已创建！")
except Exception as e:
    print(f"{e = }")