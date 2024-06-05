from pathlib import Path


file = Path('父级文件夹/abc.txt')

# 先创建父级文件夹
parent_dir = Path('父级文件夹')
parent_dir.mkdir(exist_ok=True)

# 创建这个文件
file.touch(exist_ok=True)

# 将文件进行重命名
new_name = file.parent.joinpath('新名字.txt')
file = file.rename(new_name)  # 💡 需要接受返回值，否则还是原来的路径

# 判断这个文件是否存在
print(f"重命名是否成功 -> {file.exists()}")

# 💡 .rename()方法也可以用于移动文件
target_dir = Path('新的文件夹')
target_dir.mkdir(exist_ok=True)

# 开始移动
target_path = target_dir.joinpath(file.name)
file = file.rename(target_path)
print(f"移动文件是否成功 -> {file.exists()}")


def delete_dir(folder: Path, del_content=False, verbose=False) -> bool:
    """使用Path类删除文件夹

    Args:
        folder (Path): 文件夹路径（Path实例化对象）
        del_content (bool, optional): 是否要删除有内容的文件夹. Defaults to False.

    Returns:
        bool: 是否删除成功
    """
    # 检查文件夹是否存在且为目录
    if folder.exists() and folder.is_dir():
        # 如果需要删除内容，则遍历并删除所有内容
        if del_content:
            # 遍历路径下的所有内容
            for item in folder.iterdir():
                # 如果是文件夹，则递归调用
                if item.is_dir():
                    delete_dir(item, del_content=True)
                # 如果是文件则直接删除
                else:
                    try:
                        item.unlink()
                        print(f"[INFO] 文件 {item} 已被删除") if verbose else ...
                    except FileNotFoundError:
                        print(f"[⚠️ WARNING] 文件 {item} 不存在，可能已被其他程序删除")
        
        # 尝试删除空文件夹
        try:
            folder.rmdir()
            return True
        except Exception as e:
            print(f"[❌ ERROR] 删除文件夹 {folder} 失败：{e}")
            return False
    else:
        print(f"[⚠️ WARNING] 路径不存在或者不是文件夹!")
        return False


# 删除掉这两个文件夹
print(f"删除文件夹是否成功 -> {delete_dir(parent_dir, del_content=True)}")
print(f"删除文件夹是否成功 -> {delete_dir(target_dir, del_content=True, verbose=True)}")