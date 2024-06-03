# 1. `pathlib库`介绍

相比常用的 `os.path` 而言，`pathlib` 对于目录路径的操作更简洁也更贴近 Pythonic（Python代码风格的）。但是**它不单纯是为了简化操作，还有更大的用途**。

`pathlib` 是Python内置库，Python 文档给它的定义是：

```
The pathlib module – object-oriented filesystem paths(面向对象的文件系统路径)
```

`pathlib` 提供**表示文件系统路径的类**，其语义**适用于不同的操作系统**。

<a></a>
<div align=center>
    <img src=https://img-blog.csdnimg.cn/4f3f1c68ea734e3987d101804d1e220b.png
    width=90%>
    <center></center>
</div></br>

# 2. `pathlib`库下`Path`类的基本使用

## 2.1 Path类的方法和属性

| 性质  | 用法              | 说明                                             |
| :---: | :---------------- | :----------------------------------------------- |
| 🧊方法 | absolute()        | 返回路径的绝对版本。                             |
| 🛠️属性 | anchor            | 路径的“锚”，通常是驱动器或UNC共享。              |
| 🧊方法 | as_posix()        | 返回路径的POSIX风格字符串表示。                  |
| 🧊方法 | as_uri()          | 返回路径的文件URI表示。                          |
| 🧊方法 | chmod()           | 改变文件的模式和权限位。                         |
| 🧊方法 | cwd()             | 返回当前工作目录。                               |
| 🛠️属性 | drive             | 返回路径的驱动器字母（如果有）。                 |
| 🧊方法 | exists()          | 如果路径指向一个已存在的文件或目录，返回`True`。 |
| 🧊方法 | expanduser()      | 展开路径中的`~`和`~user`。                       |
| 🧊方法 | glob()            | 返回与模式匹配的文件列表。                       |
| 🧊方法 | group()           | 返回文件组。                                     |
| 🧊方法 | home()            | 返回当前用户的主目录。                           |
| 🧊方法 | is_absolute()     | 如果路径是绝对路径，返回`True`。                 |
| 🧊方法 | is_block_device() | 如果路径指向块设备，返回`True`。                 |
| 🧊方法 | is_char_device()  | 如果路径指向字符设备，返回`True`。               |
| 🧊方法 | is_dir()          | 如果路径指向一个目录，返回`True`。               |
| 🧊方法 | is_fifo()         | 如果路径指向命名管道（FIFO），返回`True`。       |
| 🧊方法 | is_file()         | 如果路径指向一个普通文件，返回`True`。           |
| 🧊方法 | is_mount()        | 如果路径是一个挂载点，返回`True`。               |
| 🧊方法 | is_reserved()     | 如果路径是一个保留位置，返回`True`。             |
| 🧊方法 | is_socket()       | 如果路径指向一个Unix域套接字，返回`True`。       |
| 🧊方法 | is_symlink()      | 如果路径是一个符号链接，返回`True`。             |
| 🧊方法 | iterdir()         | 迭代目录中的所有路径。                           |
| 🧊方法 | joinpath()        | 连接两个或多个路径。                             |
| 🧊方法 | lchmod()          | 似`chmod`，但作用于符号链接本身。                |
| 🧊方法 | link_to()         | 创建指向此路径的硬链接。                         |
| 🧊方法 | lstat()           | 似`stat`，但作用于符号链接本身。                 |
| 🧊方法 | match()           | 使用shell样式模式匹配路径。                      |
| 🧊方法 | mkdir()           | 创建目录。                                       |
| 🛠️属性 | name              | 返回路径的最后一部分。                           |
| 🧊方法 | open()            | 打开路径指向的文件。                             |
| 🧊方法 | owner()           | 返回文件所有者。                                 |
| 🛠️属性 | parent            | 返回路径的父级目录。                             |
| 🛠️属性 | parents           | 返回所有上级目录的列表。                         |
| 🛠️属性 | parts             | 返回路径的组成部分。                             |
| 🧊方法 | read_bytes()      | 以字节的方式读取文件内容。                       |
| 🧊方法 | read_text()       | 以文本的方式读取文件内容。                       |
| 🧊方法 | relative_to()     | 计算相对路径。                                   |
| 🧊方法 | rename()          | 重命名文件或目录。                               |
| 🧊方法 | replace()         | 重命名文件或目录，即使目标已存在。               |
| 🧊方法 | resolve()         | 返回路径的绝对版本，并解析任何符号链接。         |
| 🧊方法 | rglob()           | 类似`glob`，但递归地匹配所有子目录。             |
| 🧊方法 | rmdir()           | 删除目录。                                       |
| 🛠️属性 | root              | 返回路径的根部分。                               |
| 🧊方法 | samefile()        | 如果两个路径指向相同的文件或目录，返回`True`。   |
| 🧊方法 | stat()            | 获取路径的统计信息。                             |
| 🛠️属性 | stem              | 返回没有后缀的文件名部分。                       |
| 🛠️属性 | suffix            | 返回文件扩展名。                                 |
| 🛠️属性 | suffixes          | 返回文件所有后缀的列表。                         |
| 🧊方法 | symlink_to()      | 创建指向此路径的符号链接。                       |
| 🧊方法 | touch()           | 创建一个文件。                                   |
| 🧊方法 | unlink()          | 删除文件或符号链接。                             |
| 🧊方法 | with_name()       | 返回一个新的路径，其名称部分替换为给定名称。     |
| 🧊方法 | with_suffix()     | 返回一个新的路径，其后缀替换为                   |


## 2.1 获取文件名

```python
from pathlib import Path  # 导入pathlib的Path类
import os

path = "/home/leovin/JupyterNotebookFolders/pathlib库的使用.ipynb"

p = Path(path)
print(f"获取文件名：{p.name}")  # 获取文件名：pathlib库的使用.ipynb
```

## 2.2 获取文件前缀和后缀 —— `.stem` & `.suffix`
```python
from pathlib import Path
import os

path = "/home/leovin/JupyterNotebookFolders/pathlib库的使用.ipynb"

p = Path(path)
print(f"获取前缀：{p.stem}")  # 获取前缀：pathlib库的使用
print(f"获取后缀：{p.suffix}")  # 获取后缀：.ipynb
```

## 2.3 获取文件的文件夹及上一级、上上级文件夹 —— `.parent`
```python
from pathlib import Path
import os

path = "/home/leovin/JupyterNotebookFolders/pathlib库的使用.ipynb"

p = Path(path)
print(f"获取当前文件所属文件夹：{p.parent}")
print(f"获取上一级文件夹：{p.parent.parent}")
print(f"获取上上一级文件夹：{p.parent.parent.parent}")

"""
获取当前文件所属文件夹：/home/leovin/JupyterNotebookFolders
获取上一级文件夹：/home/leovin
获取上上一级文件夹：/home
"""
```
Note: 最上级的文件夹是一个`/`:joy:

## 2.4 获取该文件所属的文件夹及其父文件夹 —— `.parents`
```python
from pathlib import Path
import os

path = "/home/leovin/JupyterNotebookFolders/pathlib库的使用.ipynb"

p = Path(path)
print(f"获取当前文件所属文件夹及其父文件夹：{p.parents}\n")

# 遍历
for idx, folder_path in enumerate(p.parents):
    print(f"No.{idx}: {folder_path}")

"""
获取当前文件所属文件夹及其父文件夹：<PosixPath.parents>

No.0: /home/leovin/JupyterNotebookFolders
No.1: /home/leovin
No.2: /home
No.3: /
"""
```

## 2.5 文件绝对路径按照`/`进行分割 —— `.parts`
```python
from pathlib import Path
import os

path = "/home/leovin/JupyterNotebookFolders/pathlib库的使用.ipynb"

p = Path(path)
print(f"将文件的绝对路径按照`/`进行分割，返回一个tuple：{p.parts}\n")

# 遍历
for idx, element in enumerate(p.parts):
    print(f"No.{idx}: {element}")

"""
获取当前文件所属文件夹及其父文件夹：<PosixPath.parents>

No.0: /
No.1: home
No.2: leovin
No.3: JupyterNotebookFolders
No.4: pathlib库的使用.ipynb
"""
```

## 2.6 获取当前工作目录 —— `Path.cwd()`
```python
from pathlib import Path
import os

path_1 = Path.cwd()
path_2 = os.getcwd()

print(f"Path.cwd(): {path_1}")
print(f"os.getcwd(): {path_2}")

"""
Path.cwd(): /home/leovin/JupyterNotebookFolders
os.getcwd(): /home/leovin/JupyterNotebookFolders
"""
```

## 2.7 获取用户`home`目录路径 —— `Path.home()`系列
```python
from pathlib import Path

print(f"获取用户home路径: {Path.home()}")

"""
获取用户home路径: /home/leovin
"""
```

## 2.8 获取文件详细信息 —— `object.stat()`
```python
from pathlib import Path

p = Path("/home/leovin/JupyterNotebookFolders/pathlib库的使用.ipynb")
print(f"获取文件详细信息: {p.stat()}\n")
print(f"获取文件字节大小: {p.stat().st_size}\n")
print(f"获取文件创建时间: {p.stat().st_ctime}\n")  # c: create
print(f"获取文件上次修改时间: {p.stat().st_mtime}\n")  # m:: modify

"""
获取文件详细信息: os.stat_result(st_mode=33204, st_ino=23860358, st_dev=2052, st_nlink=1, st_uid=1000, st_gid=1000, st_size=12125, st_atime=1653039478, st_mtime=1653039478, st_ctime=1653039478)

获取文件字节大小: 12125

获取文件创建时间: 1653039478.703161

获取文件上次修改时间: 1653039478.703161
"""
```
> 可以看到，`object.stat().xxx`就是在调用attr :joy:

## 2.9 检查目录或者文件是否存在 —— `Path("路径").exists()`
```python
from pathlib import Path

print(f"目标路径的文件是否存在: {Path('/home/leovin/JupyterNotebookFolders/xxx').exists()}")  # False
print(f"目标路径的文件是否存在: {Path('/home/leovin/JupyterNotebookFolders').exists()}")  # True
```

## 2.10 检查指定指定路径是否为folder或者file —— `Path("路径").is_file()` & `Path("路径").is_dir()`
```python
from pathlib import Path

print(f"目标路径是否为文件: {Path('/home/leovin/JupyterNotebookFolders/pathlib库的使用.ipynb').is_file()}")  # True
print(f"目标路径是否为文件夹: {Path('/home/leovin/JupyterNotebookFolders/').is_dir()}")  # True
```

> directory为文件夹

## 2.11 将相对路径转换为绝对路径 —— `Path("路径").resolve()`
```python
from pathlib import Path

print(f"转换前的路径为: {Path('./pathlib库的使用.ipynb')}")
print(f"转换后的路径为: {Path('./pathlib库的使用.ipynb').resolve()}")

"""
转换前的路径为: pathlib库的使用.ipynb
转换后的路径为: /home/leovin/JupyterNotebookFolders/pathlib库的使用.ipynb
"""
```

## 2.12 遍历一个目录 —— `Path("路径").iterdir()`
```python
from pathlib import Path

path_object = Path("/home/leovin/JupyterNotebookFolders/").iterdir()
print(f"path_object: {path_object}")  # path_object: <generator object Path.iterdir at 0x7f0ca0061c10>

# 迭代目录对象
for idx, element in enumerate(path_object):
    print(f"No.{idx}: {element}")
    
"""
path_object: <generator object Path.iterdir at 0x7f0ca0061c10>
No.0: /home/leovin/JupyterNotebookFolders/temp
No.1: /home/leovin/JupyterNotebookFolders/torch.meshgrid().ipynb
No.2: /home/leovin/JupyterNotebookFolders/array.argsort().ipynb
No.3: /home/leovin/JupyterNotebookFolders/切片....ipynb
No.4: /home/leovin/JupyterNotebookFolders/logging信息.log
No.5: /home/leovin/JupyterNotebookFolders/Python中类的私有变量、私有方法、静态方法.ipynb
No.6: /home/leovin/JupyterNotebookFolders/temp_file.ipynb
No.7: /home/leovin/JupyterNotebookFolders/example.log
No.8: /home/leovin/JupyterNotebookFolders/Python语法.ipynb
No.9: /home/leovin/JupyterNotebookFolders/.ipynb_checkpoints
No.10: /home/leovin/JupyterNotebookFolders/craw.data.txt
No.11: /home/leovin/JupyterNotebookFolders/test.txt
No.12: /home/leovin/JupyterNotebookFolders/logging的学习.ipynb
No.13: /home/leovin/JupyterNotebookFolders/Test.ipynb
No.14: /home/leovin/JupyterNotebookFolders/算法题
No.15: /home/leovin/JupyterNotebookFolders/functions.ipynb
No.16: /home/leovin/JupyterNotebookFolders/mask转0&1.ipynb
No.17: /home/leovin/JupyterNotebookFolders/Multi_Task_in_Python.ipynb
No.18: /home/leovin/JupyterNotebookFolders/pathlib库的使用.ipynb
No.19: /home/leovin/JupyterNotebookFolders/三种激活函数绘制.ipynb
"""
```

## 2.13 获取所有符合pattern的文件 —— `Path("路径").glob("folder1/xxx.格式")`
```python
from pathlib import Path

pattern = "JupyterNotebookFolders/*.ipynb"
glob_generator = Path("/home/leovin/").glob(pattern)

# 遍历返回的对象 -> 返回的是绝对路径
for idx, element in enumerate(glob_generator):
    print(f"No.{idx}: {element}")
    
"""
No.0: /home/leovin/JupyterNotebookFolders/torch.meshgrid().ipynb
No.1: /home/leovin/JupyterNotebookFolders/array.argsort().ipynb
No.2: /home/leovin/JupyterNotebookFolders/切片....ipynb
No.3: /home/leovin/JupyterNotebookFolders/Python中类的私有变量、私有方法、静态方法.ipynb
No.4: /home/leovin/JupyterNotebookFolders/temp_file.ipynb
No.5: /home/leovin/JupyterNotebookFolders/Python语法.ipynb
No.6: /home/leovin/JupyterNotebookFolders/logging的学习.ipynb
No.7: /home/leovin/JupyterNotebookFolders/Test.ipynb
No.8: /home/leovin/JupyterNotebookFolders/functions.ipynb
No.9: /home/leovin/JupyterNotebookFolders/mask转0&1.ipynb
No.10: /home/leovin/JupyterNotebookFolders/Multi_Task_in_Python.ipynb
No.11: /home/leovin/JupyterNotebookFolders/pathlib库的使用.ipynb
No.12: /home/leovin/JupyterNotebookFolders/三种激活函数绘制.ipynb
"""
```

## 2.14 删除文件（非目录）—— `Path("路径").unlink()`
```python
from pathlib import Path

# 当前文件夹下的txt文件
for idx, element in enumerate(Path("./").glob("*.txt")):
    print(f"No.{idx}: {element}")
    
print("-" * 30)

"""
    删除指定的文件（非目录）
        1. 是真的删除而非unlink
        2. 如果文件不存在则保存
"""
try:
    Path("./will_be_deleted.txt").unlink()
except Exception as e:
    print(f"删除文件发生错误，原因为: {e}")

# 当前文件夹下的txt文件
for idx, element in enumerate(Path("./").glob("*.txt")):
    print(f"No.{idx}: {element}")

"""
No.0: will_be_deleted.txt
No.1: craw.data.txt
No.2: test.txt
------------------------------
No.0: craw.data.txt
No.1: test.txt


报错的输出：
No.0: craw.data.txt
No.1: test.txt
------------------------------
删除文件发生错误，原因为: [Errno 2] No such file or directory: 'will_be_deleted.txt'
No.0: craw.data.txt
No.1: test.txt
"""
```



# 3. `pathlib`库与`os`库的对比
|`os`库|`pathlib`库|描述|英文说明|
|--|--|--|--|
|`os.path.abspath(("文件路径"))`|`Path("文件路径").resolve()`|将路径转换为绝对路径|-|
|`os.chmod()`|`Path("文件路径").chmod(xxx)`|更改文件权限|change mode|
|`os.mkdir(("文件路径"))`|`Path("文件路径").mkdir()`|新建文件夹|make directory|
|`os.rename("文件路径",  "xxx")`|`Path("文件路径").rename("xxx")`|重命名文件/文件夹名称|-|
|`os.replace(a, b)`|`Path("文件路径").replace(a, b)`|替换字符串|-|
|`os.rmdir()`|`Path("文件路径").rmdir()`|删除文件夹（里面必须是空的）|remove directory|
|`os.remove("文件路径") / os.unlink("文件路径")`|`Path("文件路径").unlink()`|删除文件（非目录）|-|
|`os.getcwd()`|`Path("文件路径").cwd()`|获取当前文件工作目录|current work directory|
|`os.path.isdir()`|`Path("文件路径").is_dir()`|判断当前路径是否为目录|-|
|`os.path.isfile()`|`Path("文件路径").is_file()`|判断当前路径是否为文件|-|
|`os.stat()`|`Path("文件路径").stat()`|返回当前路径的信息|status|
|`os.path.isabs()`|`Path("文件路径").is_absolute()`|判断当前路径是否为绝对路径|-|
|`os.path.basename()`|`Path("文件路径").name`|返回文件/目录的基础名称（不带路径）|-|
|`os.path.dirname()`|`Path("文件路径").parent`|返回路径所属文件夹名称|-|
|`os.path.samefile()`|`Path("文件路径").samefile(xxx)`|判断两个文件是否相同|-|
|`os.path.splitext("文件路径")`|`(Path("文件路径").stem, Path("文件路径").suffix)`|将文件名分离，分成前缀和后缀|stem + suffix|

# 参考
1. https://www.jb51.net/article/193402.htm
2. https://docs.python.org/zh-cn/3/library/pathlib.html?highlight=pathlib#module-pathlib
