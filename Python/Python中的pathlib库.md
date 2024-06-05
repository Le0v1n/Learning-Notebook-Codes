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
    width=50%>
    <center></center>
</div></br>

# 2. `pathlib`库下`Path`类的基本使用

## 2.1 Path类的属性和方法

| 性质  | 用法 | 结果 | 数据类型 | 说明 |
| :---: | :--- | :--- | :------- | :--- |
|| p| /mnt/f/Learning-Notebook-Codes/Datasets/coco128.tar.gz| <class 'pathlib.PosixPath'> | Path的实例化对象|
||||||
| 🛠️  属性 | p.anchor| /| <class 'str'>| 路径的“锚”，通常是驱动器或UNC共享|
| 🛠️  属性 | p.drive|| <class 'str'>| 返回路径的驱动器字母（如果有）|
| 🛠️  属性 | p.name| coco128.tar.gz| <class 'str'>| 返回路径的最后一部分|
| 🛠️  属性 | p.parent| /mnt/f/Learning-Notebook-Codes/Datasets| <class 'pathlib.PosixPath'> | 返回路径的父级目录（💡 还是一个Path对象）|
| 🛠️  属性 | p.parts| ('/', 'mnt', 'f', 'Learning-Notebook-Codes', 'Datasets', 'coco128.tar.gz')| <class 'tuple'>| 返回路径的组成部分|
||||||
| 🛠️  属性 | p.root|/| <class 'str'>| 返回路径的根部分（💡 如果是相对路径则为""）：|
| 🛠️  属性 | p.stem| coco128.tar| <class 'str'>| 返回没有后缀的文件名部分|
| 🛠️  属性 | p.suffix| .gz| <class 'str'>| 返回文件扩展名|
| 🛠️  属性 | p.suffixes| ['.tar', '.gz']| <class 'list'>| 返回文件所有后缀的列表|
||||||
| 🧊 方法 | p.absolute()| /mnt/f/Learning-Notebook-Codes/Datasets/coco128.tar.gz| <class 'pathlib.PosixPath'> | 返回对象的绝对路径|
| 🧊 方法 | p.as_posix()| /mnt/f/Learning-Notebook-Codes/Datasets/coco128.tar.gz| <class 'str'>| 返回路径的POSIX风格字符串表示|
| 🧊 方法 | p.as_uri()| file:///mnt/f/Learning-Notebook-Codes/Datasets/coco128.tar.gz| <class 'str'>| 返回路径的文件URI表示（💡 如果创建p为相对路径则报错）|
| 🧊 方法 | p.chmod(0o744)| None| <class 'NoneType'>| 改变文件的模式和权限位（💡 如果文件不存在则报错）|
| 🧊 方法 | p.cwd()| /mnt/f/Learning-Notebook-Codes| <class 'pathlib.PosixPath'> | 返回当前工作目录（绝对路径）|
| 🧊 方法 | p.expanduser()| /mnt/f/Learning-Notebook-Codes/Datasets/coco128.tar.gz| <class 'pathlib.PosixPath'> | 展开路径中的~和~user|
| 🧊 方法 | p.home()| /home/leovin| <class 'pathlib.PosixPath'> | 返回当前用户的主目录|
| 🧊 方法 | p.is_absolute()| True| <class 'bool'>| 判断当前路径是否为绝对路径|
| 🧊 方法 | p.is_dir()| False| <class 'bool'>| 判断当前路径是否为一个文件夹📂|
| 🧊 方法 | p.is_file()| True| <class 'bool'>| 判断当前路径是否为一个文件📑|
| 🧊 方法 | [dir.name for dir in list(d.iterdir())] | ['images', 'labels', 'labels.cache']| <class 'generator'>| 迭代目录中的所有路径（💡 如果不是一个目录则报错）|
| 🧊 方法 | d.join(str, str)| Datasets/coco128/val/123/abc/结束| <class 'pathlib.PosixPath'> | 连接两个或多个路径|
| 🧊 方法 | d.mkdir()| None| <class 'NoneType'>| 创建目录（💡 有两个报错参数！）|
| 🧊 方法 | f.relative_to(base_path)| train/labels/000000000572.txt| <class 'pathlib.PosixPath'> | 计算相对路径（💡 需提供基准路径）|
| 🧊 方法 | p.rename('Datasets/ms-coco128.tar.gz')  | Datasets/ms-coco128.tar.gz| <class 'pathlib.PosixPath'> | 重命名文件或目录|
| 🧊 方法 | p.resolve()| /mnt/f/Learning-Notebook-Codes/Datasets/coco128.tar.gz| <class 'pathlib.PosixPath'> | 返回路径的绝对版本，并解析任何符号链接|
| 🧊 方法 | Path('Datasets/empty_dir').rmdir()| None| <class 'NoneType'>| 删除目录（💡 目录不为空或不是目录，会报错）|
| 🧊 方法 | p.samefile(p2)| True| <class 'bool'>| 如果两个路径指向相同的文件或目录，返回True|
| 🧊 方法 | p.stat()| os.stat_result(st_mode=33279, st_ino=3940649674488502, st_dev=49, st_nlink=1, st_uid=1000, st_gid=1000, st_size=6909053, st_atime=1717463716, st_mtime=1717463716, st_ctime=1717468441) | <class 'os.stat_result'>    | 获取路径的统计信息|
| 🧊 方法 | p.touch(mode: int = 438, exist_ok: bool = True)| None| <class 'NoneType'>| 创建一个文件（💡 1. 不能创建文件夹 2.需要确保父目录存在）|
| 🧊 方法 | p.with_name(name='新名字')| /mnt/f/Learning-Notebook-Codes/Datasets/ms-coco128.zip| <class 'pathlib.PosixPath'> | 返回一个新的路径，其名称部分替换为给定名称（💡 需要我们指定后缀） |
| 🧊 方法 | f.with_name(suffix='新后缀')| Datasets/coco128/train/labels/000000000572.zip| <class 'pathlib.PosixPath'> | 返回一个新的路径，其后缀部分替换为给定后缀（💡 不能缺少.）|

## 2.1 🛠️ 属性解析

### .name：返回路径的最后一部分

- 作用：返回路径的最后一部分
- 示例代码：

```python
from pathlib import Path


dataset = Path('/mnt/f/Learning-Notebook-Codes/Datasets')
dataset_compressed_package = Path('/mnt/f/Learning-Notebook-Codes/Datasets.tar.gz')
image = Path('Datasets/coco128/train/images/000000000061.jpg')
label = Path('Datasets/coco128/train/labels/000000000061.txt')

print(f"{dataset.name = }")
print(f"{dataset_compressed_package.name = }")
print(f"{image.name = }")
print(f"{label.name = }")
```

```
dataset.name = 'Datasets'
dataset_compressed_package.name = 'Datasets.tar.gz'
image.name = '000000000061.jpg'
label.name = '000000000061.txt'
```

### .stem和.suffix：获取文件前缀和后缀

- 作用：
  - `.stem`：返回没有后缀的文件名部分
  - `.suffix`：返回文件扩展名
- 💡 注意：对于 `.tar.gz` 这样的双后缀的文件，只会返回最后一个后缀
- 示例代码：


```python
from pathlib import Path


dataset = Path('/mnt/f/Learning-Notebook-Codes/Datasets')
dataset_compressed_package = Path('/mnt/f/Learning-Notebook-Codes/Datasets.tar.gz')
image = Path('Datasets/coco128/train/images/000000000061.jpg')
label = Path('Datasets/coco128/train/labels/000000000061.txt')

print(f"[前缀-{dataset.name}] {dataset.stem}")
print(f"[后缀-{dataset.name}] {dataset.suffix}")
print(f"[前缀-{dataset_compressed_package.name}] {dataset_compressed_package.stem}")
print(f"[后缀-{dataset_compressed_package.name}] {dataset_compressed_package.suffix}")
print(f"[前缀-{image.name}] {image.stem}")
print(f"[后缀-{image.name}] {image.suffix}")
print(f"[前缀-{label.name}] {label.stem}")
print(f"[后缀-{label.name}] {label.suffix}")
```

```
[前缀-Datasets] Datasets
[后缀-Datasets] 
[前缀-Datasets.tar.gz] Datasets.tar  # 💡 .tar也在前缀中
[后缀-Datasets.tar.gz] .gz           # 💡 只会返回.gz这一个后缀（即只返回最后一个后缀）
[前缀-000000000061.jpg] 000000000061
[后缀-000000000061.jpg] .jpg
[前缀-000000000061.txt] 000000000061
[后缀-000000000061.txt] .txt
```

### .parent：返回路径的父级目录

- 作用：返回路径的父级目录
- 💡 注意：
  - 返回的仍然是一个 `Path` 对象
  - 可以连续调用 `.parent` 属性
  - 如果 Path 对象创建时用的<font color='red'><b>绝对路径</b></font>，最终的父目录是 `/`
  - 如果 Path 对象创建时用的<font color='blue'><b>相对路径</b></font>，最终的父目录是 `.`
- 示例代码：

```python
from pathlib import Path


def print_parents(p: Path):
    parents = p.parents
    print(f"{type(parents) = }")
    for i, dirpath in enumerate(parents):
        print(f"[{p.name}] [L{i}] {dirpath}")
    print('-'*100)


dataset = Path('/mnt/f/Learning-Notebook-Codes/Datasets')
dataset_compressed_package = Path('/mnt/f/Learning-Notebook-Codes/Datasets.tar.gz')
image = Path('Datasets/coco128/train/images/000000000061.jpg')
label = Path('Datasets/coco128/train/labels/000000000061.txt')

print_parents(dataset)
print_parents(dataset_compressed_package)
print_parents(image)
print_parents(label)
```

```
dataset.parent = PosixPath('/mnt/f/Learning-Notebook-Codes')
dataset.parent.parent = PosixPath('/mnt/f')
dataset.parent.parent.parent = PosixPath('/mnt')
dataset.parent.parent.parent.parent = PosixPath('/')
dataset.parent.parent.parent.parent.parent = PosixPath('/')
----------------------------------------------------------------------------------------------------
dataset_compressed_package.parent = PosixPath('/mnt/f/Learning-Notebook-Codes')
dataset_compressed_package.parent.parent = PosixPath('/mnt/f')
dataset_compressed_package.parent.parent.parent = PosixPath('/mnt')
dataset_compressed_package.parent.parent.parent.parent = PosixPath('/')
dataset_compressed_package.parent.parent.parent.parent.parent = PosixPath('/')
----------------------------------------------------------------------------------------------------
image.parent = PosixPath('Datasets/coco128/train/images')
image.parent.parent = PosixPath('Datasets/coco128/train')
image.parent.parent.parent = PosixPath('Datasets/coco128')
image.parent.parent.parent.parent = PosixPath('Datasets')
image.parent.parent.parent.parent.parent = PosixPath('.')
----------------------------------------------------------------------------------------------------
label.parent = PosixPath('Datasets/coco128/train/labels')
label.parent.parent = PosixPath('Datasets/coco128/train')
label.parent.parent.parent = PosixPath('Datasets/coco128')
label.parent.parent.parent.parent = PosixPath('Datasets')
label.parent.parent.parent.parent.parent = PosixPath('.')
```

### .parents：获取所有的父级目录

- 作用：返回路径的父级目录
- 💡 注意：
  - 返回的仍然是一个`pathlib._PathParents`对象，它是一个可迭代对象
  - 如果 Path 对象创建时用的<font color='red'><b>绝对路径</b></font>，最终的父目录是 `/`
  - 如果 Path 对象创建时用的<font color='blue'><b>相对路径</b></font>，最终的父目录是 `.`
- 示例代码：

```python
from pathlib import Path


def print_parents(p: Path):
    parents = p.parents
    print(f"[{p.name}] {type(parents) = }")
    for i, dirpath in enumerate(p.parents):
        print(f"[{p.name}] [L{i}] {dirpath}")
    print('-'*100)


dataset = Path('/mnt/f/Learning-Notebook-Codes/Datasets')
dataset_compressed_package = Path('/mnt/f/Learning-Notebook-Codes/Datasets.tar.gz')
image = Path('Datasets/coco128/train/images/000000000061.jpg')
label = Path('Datasets/coco128/train/labels/000000000061.txt')

print_parents(dataset)
print_parents(dataset_compressed_package)
print_parents(image)
print_parents(label)
```

```
[Datasets] type(parents) = <class 'pathlib._PathParents'>
[Datasets] [L0] /mnt/f/Learning-Notebook-Codes
[Datasets] [L1] /mnt/f
[Datasets] [L2] /mnt
[Datasets] [L3] /
----------------------------------------------------------------------------------------------------
[Datasets.tar.gz] type(parents) = <class 'pathlib._PathParents'>
[Datasets.tar.gz] [L0] /mnt/f/Learning-Notebook-Codes
[Datasets.tar.gz] [L1] /mnt/f
[Datasets.tar.gz] [L2] /mnt
[Datasets.tar.gz] [L3] /
----------------------------------------------------------------------------------------------------
[000000000061.jpg] type(parents) = <class 'pathlib._PathParents'>
[000000000061.jpg] [L0] Datasets/coco128/train/images
[000000000061.jpg] [L1] Datasets/coco128/train
[000000000061.jpg] [L2] Datasets/coco128
[000000000061.jpg] [L3] Datasets
[000000000061.jpg] [L4] .
----------------------------------------------------------------------------------------------------
[000000000061.txt] type(parents) = <class 'pathlib._PathParents'>
[000000000061.txt] [L0] Datasets/coco128/train/labels
[000000000061.txt] [L1] Datasets/coco128/train
[000000000061.txt] [L2] Datasets/coco128
[000000000061.txt] [L3] Datasets
[000000000061.txt] [L4] .
----------------------------------------------------------------------------------------------------
```

### .parts：返回路径的组成部分

- 作用：返回路径的组成部分
- 💡 注意：
  - 返回的是一个`tuple`对象。
- 示例代码：

```python
from pathlib import Path


dataset = Path('/mnt/f/Learning-Notebook-Codes/Datasets')
dataset_compressed_package = Path('/mnt/f/Learning-Notebook-Codes/Datasets.tar.gz')
image = Path('Datasets/coco128/train/images/000000000061.jpg')
label = Path('Datasets/coco128/train/labels/000000000061.txt')

print(f"[{dataset.name}] {dataset.parts}")
print(f"[{dataset_compressed_package.name}] {dataset_compressed_package.parts}")
print(f"[{image.name}] {image.parts}")
print(f"[{label.name}] {label.parts}")
```

```
[    Datasets    ] ('/', 'mnt', 'f', 'Learning-Notebook-Codes', 'Datasets')
[Datasets.tar.gz ] ('/', 'mnt', 'f', 'Learning-Notebook-Codes', 'Datasets.tar.gz')
[000000000061.jpg] ('Datasets', 'coco128', 'train', 'images', '000000000061.jpg')
[000000000061.txt] ('Datasets', 'coco128', 'train', 'labels', '000000000061.txt')
```

## 2.2 🧊 方法解析

### 🧊 [1] Path.cwd()：获取当前工作目录

- 作用：根据当前对象，返回当前工作目录。
- 💡 注意：
  - 如果直接调用`Path.cwd()`返回是一个`Path`对象，如需获取字符串，则需要`Path.cwd().name`
  - 如果通过对象调用`p.cwd()`返回是一个字符串，而非`Path`对象
- 示例代码：

```python
from pathlib import Path
import os


dataset = Path('/mnt/f/Learning-Notebook-Codes/Datasets')
dataset_compressed_package = Path('/mnt/f/Learning-Notebook-Codes/Datasets.tar.gz')
image = Path('Datasets/coco128/train/images/000000000061.jpg')
label = Path('Datasets/coco128/train/labels/000000000061.txt')

print(f"[使用os库] {os.getcwd() = }")
print(f"[直接调用] {Path.cwd() = }")
print(f"[直接调用] {Path.cwd().name = }")
print(f"[{dataset.name}] {dataset.cwd()}")
print(f"[{dataset_compressed_package.name}] {dataset_compressed_package.cwd()}")
print(f"[{image.name}] {image.cwd()}")
print(f"[{label.name}] {label.cwd()}")
```

```
[使用os库] os.getcwd() = '/mnt/f/Learning-Notebook-Codes'
[直接调用] Path.cwd() = PosixPath('/mnt/f/Learning-Notebook-Codes')
[直接调用] Path.cwd().name = 'Learning-Notebook-Codes'
[Datasets] /mnt/f/Learning-Notebook-Codes
[Datasets.tar.gz] /mnt/f/Learning-Notebook-Codes
[000000000061.jpg] /mnt/f/Learning-Notebook-Codes
[000000000061.txt] /mnt/f/Learning-Notebook-Codes
```

### 🧊 [2] Path.home()：返回当前用户的家目录

- 作用：返回当前用户的家目录，即`/user/home`
- 💡 注意：
  - 如果直接调用`Path.home()`返回是一个`Path`对象，如需获取字符串，则需要`.name`
  - 如果通过对象调用`p.home()`返回是一个字符串，而非`Path`对象
- 示例代码：

```python
from pathlib import Path


dataset = Path('/mnt/f/Learning-Notebook-Codes/Datasets')
dataset_compressed_package = Path('/mnt/f/Learning-Notebook-Codes/Datasets.tar.gz')
image = Path('Datasets/coco128/train/images/000000000061.jpg')
label = Path('Datasets/coco128/train/labels/000000000061.txt')

print(f"[直接调用] {Path.home() = }")
print(f"[直接调用] {Path.home().name = }")
print(f"[{dataset.name}] {dataset.home()}")
print(f"[{dataset_compressed_package.name}] {dataset_compressed_package.home()}")
print(f"[{image.name}] {image.home()}")
print(f"[{label.name}] {label.home()}")
```

```
[直接调用] Path.home() = PosixPath('/home/leovin')
[直接调用] Path.home().name = 'leovin'
[Datasets] /home/leovin
[Datasets.tar.gz] /home/leovin
[000000000061.jpg] /home/leovin
[000000000061.txt] /home/leovin
```

### 🧊 [3] object.stat()：获取文件详细信息

在这个例子中，我们首先创建了一个指向名为 `example.txt` 的文件的 `Path` 对象。然后，我们调用 `stat()` 方法来获取文件的统计信息，并将其存储在 `stat_info` 变量中。接着，我们打印出文件的大小、最后修改时间和文件所有者的用户ID。
请注意，`st_atime`、`st_mtime` 和 `st_ctime` 返回的是自纪元（通常在Unix系统上是1970年1月1日）以来的秒数。为了将这些时间转换为更易读的格式，我们使用了 `time.ctime()` 函数。在实际应用中，你可能需要使用 `datetime.datetime.fromtimestamp()` 方法来获取更精确的日期和时间表示。

- 作用：获取文件详细信息，包括文件大小、创建时间、最后访问时间、最后修改时间等。该方法返回一个`os.stat_result`对象，其常用属性如下：
  - `st_mode`: 文件类型和权限信息。
  - `st_size`: 文件大小（以字节为单位）。
  - `st_atime`: 文件最后访问时间。
  - `st_mtime`: 文件最后修改时间。
  - `st_ctime`: 文件状态改变时间（Windows上为创建时间）。
  - ...（不常用）
- 💡 注意：
  - 文件大小、时间不易读，我们可以改为易读的形式。
- 示例代码：

```python
from pathlib import Path
import datetime


image = Path('Datasets/coco128/train/images/000000000061.jpg')

print(f"------------------------------ 原始信息 ------------------------------")
print(f"获取文件详细信息: {image.stat()}")
print(f"获取文件字节大小: {image.stat().st_size}")
print(f"获取文件创建时间: {image.stat().st_ctime}")  # c: create
print(f"获取文件上次修改时间: {image.stat().st_mtime}")  # m:: modify

print(f"---------------------------- 改为易读形式 ----------------------------")
# 获取文件详细统计信息
stat_info = image.stat()

# 打印文件字节大小，并转换为KB和MB
file_size_bytes = stat_info.st_size
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024
print(f"文件字节大小: {file_size_bytes} bytes, {file_size_kb:.2f} KB, {file_size_mb:.2f} MB")

# 打印文件创建时间（转换为人类可读的格式）
create_time = datetime.datetime.fromtimestamp(stat_info.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
print(f"文件创建时间: {create_time}")

# 打印文件上次修改时间（转换为人类可读的格式）
modify_time = datetime.datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
print(f"文件上次修改时间: {modify_time}")
```

```
------------------------------ 原始信息 ------------------------------
获取文件详细信息: os.stat_result(st_mode=33279, st_ino=1125899907375496, st_dev=49, st_nlink=1, 
                                 st_uid=1000, st_gid=1000, st_size=135566, st_atime=1714285132, 
                                 st_mtime=1666146712, st_ctime=1707038640)
获取文件字节大小: 135566
获取文件创建时间: 1707038640.4572322
获取文件上次修改时间: 1666146712.0
---------------------------- 改为易读形式 ----------------------------
文件字节大小: 135566 bytes, 132.39 KB, 0.13 MB
文件创建时间: 2024-02-04 17:24:00
文件上次修改时间: 2022-10-19 10:31:52
```

### 🧊 [4] .exists()：检查目录或者文件是否存在

- 作用：检查`Path`对象所指的路径是否存在（可以是📑文件也可以是📂文件夹）
- 💡 注意：
  - 等价于`os.path.exists(具体的路径)`
  - `os.path.exists(Path)`也是可以的（`Path`甚至都不用`.name`）
- 示例代码：

```python
from pathlib import Path
import os


dataset = Path('/mnt/f/Learning-Notebook-Codes/Datasets')
dataset_compressed_package = Path('/mnt/f/Learning-Notebook-Codes/Datasets.tar.gz')
image = Path('Datasets/coco128/train/images/000000000061.jpg')
label = Path('Datasets/coco128/train/labels/000000000061.txt')

print(f"[{dataset.name}] [调用os库] {os.path.exists(dataset) = }")
print(f"[{dataset.name}] {dataset.exists() = }")

print(f"[{dataset_compressed_package.name}] {dataset_compressed_package.exists() = }")
print(f"[{dataset_compressed_package.name}] [调用os库] {os.path.exists(dataset_compressed_package) = }")

print(f"[{image.name}] {image.exists() = }")
print(f"[{image.name}] [调用os库] {os.path.exists(image) = }")

print(f"[{label.name}] {label.exists() = }")
print(f"[{label.name}] [调用os库] {os.path.exists(label) = }")
```

```
[Datasets] [调用os库] os.path.exists(dataset) = True
[Datasets] dataset.exists() = True

[Datasets.tar.gz] dataset_compressed_package.exists() = False
[Datasets.tar.gz] [调用os库] os.path.exists(dataset_compressed_package) = False

[000000000061.jpg] image.exists() = True
[000000000061.jpg] [调用os库] os.path.exists(image) = True

[000000000061.txt] label.exists() = True
[000000000061.txt] [调用os库] os.path.exists(label) = True
```

### 🧊 [5] .is_file()和.is_dir()：判断路径的性质（文件/文件夹）

- 作用：判断路径的性质（文件/文件夹），返回值是一个boolean类型。
- 💡 注意：
  - 等价于`os.path.isfile(路径)和os.path.isdir(路径)`
- 示例代码：

```python
from pathlib import Path
import os


def get_path_type_Path(path: Path) -> str:
    if path.is_file():
        return 'file'
    elif path.is_dir():
        return 'dir'
    else:
        return 'unknown'
    
    
def get_path_type_os(path: str) -> str:
    if os.path.isfile(path):
        return 'file'
    elif os.path.isdir(path):
        return 'dir'
    else:
        return 'unknown'
            

dataset = Path('/mnt/f/Learning-Notebook-Codes/Datasets')
dataset_compressed_package = Path('/mnt/f/Learning-Notebook-Codes/Datasets.tar.gz')
image = Path('Datasets/coco128/train/images/000000000061.jpg')
label = Path('Datasets/coco128/train/labels/000000000061.txt')

print(f"[{dataset.name}] [调用os库] {get_path_type_os(dataset.absolute()) = }")
print(f"[{dataset.name}] {get_path_type_Path(dataset) = }")
print(f"[{dataset_compressed_package.name}] [调用os库] {get_path_type_os(dataset_compressed_package.absolute()) = }")
print(f"[{dataset_compressed_package.name}] {get_path_type_Path(dataset_compressed_package) = }")
print(f"[{image.name}] [调用os库] {get_path_type_os(image.absolute()) = }")
print(f"[{image.name}] {get_path_type_Path(image) = }")
print(f"[{label.name}] [调用os库] {get_path_type_os(label.absolute()) = }")
print(f"[{label.name}] {get_path_type_Path(label) = }")
```

```
[Datasets] [调用os库] get_path_type_os(dataset.absolute()) = 'dir'
[Datasets] get_path_type_Path(dataset) = 'dir'

[Datasets.tar.gz] [调用os库] get_path_type_os(dataset_compressed_package.absolute()) = 'unknown'
[Datasets.tar.gz] get_path_type_Path(dataset_compressed_package) = 'unknown'

[000000000061.jpg] [调用os库] get_path_type_os(image.absolute()) = 'file'
[000000000061.jpg] get_path_type_Path(image) = 'file'

[000000000061.txt] [调用os库] get_path_type_os(label.absolute()) = 'file'
[000000000061.txt] get_path_type_Path(label) = 'file'
```

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
