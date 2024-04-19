
# 0. 引言

Python 中的 os 模块和 sys 是比较常用的模块，os 全称为 Operation System，sys 全称为 System。使用这两个模块之前必须进行导入（集成于 Python，无需额外安装）：

```python
import os
import sys
```

# 1. os 模块

## 1.1 os.getcwd()

**作用**：获取当前的工作路径

> 💡  cwd: current work directory，当前工作目录

**举个例子**：文件 `exp1.py` 的文件所在路径为：`/mnt/f/Projects/项目/本地代码/Learning-Notebook-Codes/Python/code/exp1.py`，其代码内容为：

```python
import os


print(f"{os.getcwd() = }")
```

运行结果：

```
os.getcwd() = '/mnt/f/Projects/项目/本地代码/Learning-Notebook-Codes'
```

可以看到，当前工作目录就是项目目录，即 `Learning-Notebook-Codes` 这个项目的目录。

如果 `exp1.py` 是被 `exp2.py` 文件调用的，那么结果会是怎样呢？

- `exp1.py` 所在路径：`/mnt/f/Projects/项目/本地代码/Learning-Notebook-Codes/Python/code/exp1.py`
- `exp2.py` 所在路径：`/mnt/f/Projects/项目/本地代码/Learning-Notebook-Codes/Python/code/exp2.py`

`exp2.py` 的内容如下：

```python
from Python.code import exp1


print(f"{__name__} 执行完毕！")
```

执行后报错，信息如下：

```
Traceback (most recent call last):
  File "Python/code/exp2.py", line 1, in <module>
    from Python.code import exp1
ModuleNotFoundError: No module named 'Python'
```

⚠️  OBS：这是因为我使用的 IDE 是 VSCode 而不是 PyCharm，如果是 PyCharm 则不会报这个错误。为了解决这个问题，我们可以加入下面的语句：

```python
import os
import sys
sys.path.append(os.getcwd())
```

即 `exp2.py` 的内容为：

```python
import os
import sys
sys.path.append(os.getcwd())
from Python.code import exp1


print(f"{__name__} 执行完毕！")
```

此时我们再次执行就没有问题了，执行结果如下：

```
os.getcwd() = '/mnt/f/Projects/项目/本地代码/Learning-Notebook-Codes'
__main__ 执行完毕！
```

可以看到，`os.getcwd()` 在 VSCode 中就是项目的路径。

## 1.2 os.listdir(path)

**作用**：传入任意一个 `path` 路径，返回的是该路径下所有文件和文件夹组成的列表 (list)。

### 1.2.1 例子：单个文件夹下（没有子文件夹）

```
Python/code
├── exp1.py
├── exp2.py
└── exp3.py

0 directories, 3 files
```

```python
import os


files_list = os.listdir('Python/code')
print(f"{files_list = }")
```

```
files_list = ['exp1.py', 'exp2.py', 'exp3.py']
```

### 1.2.2 例子：单个文件夹（包含子文件夹）

```
Python/code
├── exp1.py
├── exp2.py
├── exp3.py
├── exp4.py
├── 📂folder1
│   ├── exp1.py
│   └── exp2.c
└── 📂folder2
    ├── exp1.md
    └── exp2.txt

2 directories, 8 files
```


```python
import os


files_list = os.listdir('Python/code')
print(f"{files_list = }")
```

```
files_list = ['exp1.py', 'exp2.py', 'exp3.py', 'exp4.py', '📂folder1', '📂folder2']
```

可以看到，`os.listdir()` 并不会遍历子文件夹！

## 1.3 os.walk(path)

**作用** ：传入任意一个路径，深层次遍历指定路径下的所有子文件夹，返回的是一个由〔路径 (dirpath)〕、〔文件夹列表 (dirnames)〕、〔文件列表 (filenames)〕组成的元组 (tuple)。

### 1.3.1 例子：没有子文件夹

```
Python/code/📂folder1
├── exp1.py
└── exp2.c

0 directories, 2 files
```

```python
import os


path = 'Python/code/📂folder1'

result = os.walk(path)

print(f"{result = }")
print(f"{type(result) = }")
```

```
result = <generator object walk at 0x7eff00992b30>
type(result) = <class 'generator'>
```

我们得到的是一个生成器对象，那我们尝试对其进行遍历：

```python
import os


path = 'Python/code/📂folder1'

result = os.walk(path)

print(f"{result = }")
print(f"{type(result) = }")

for items in result:
    print(f"{items = }")
```

```
result = <generator object walk at 0x7fdb596d6b30>
type(result) = <class 'generator'>
items = ('Python/code/📂folder1', [], ['exp1.py', 'exp2.c'])
```

可以看到，这个生成器对象的 `.__next__` 方法会返回一个 `tuple`，和我们上面说的一致，这个 `tuple` 里面存放的分别是：dirpath, dirnames, filenames。那我们修改一下遍历的样式：

```python
import os


for dirpath, dirnames, filenames in os.walk('Python/code/📂folder1'):
    print(f"{dirpath = }")
    print(f"{dirnames = }")
    print(f"{filenames = }")
```

```
dirpath = 'Python/code/📂folder1'
dirnames = []
filenames = ['exp1.py', 'exp2.c']
```

### 1.3.2 例子：单个文件夹（包含子文件夹）

```
Python/code
├── exp1.py
├── exp2.py
├── exp3.py
├── exp4.py
├── exp5.py
├── exp6.py
├── 📂folder1
│   ├── exp1.py
│   └── exp2.c
└── 📂folder2
    ├── exp1.md
    └── exp2.txt

2 directories, 10 files
```

```python
import os


iter_num = 0
for dirpath, dirnames, filenames in os.walk('Python/code'):
    print(f"-------------------------- {iter_num = } --------------------------")
    print(f"{dirpath = }")
    print(f"{dirnames = }")
    print(f"{filenames = }")
    iter_num += 1
```

```
-------------------------- iter_num = 0 --------------------------
dirpath = 'Python/code'
dirnames = ['📂folder1', '📂folder2']
filenames = ['exp1.py', 'exp2.py', 'exp3.py', 'exp4.py', 'exp5.py', 'exp6.py']
-------------------------- iter_num = 1 --------------------------
dirpath = 'Python/code/📂folder1'
dirnames = []
filenames = ['exp1.py', 'exp2.c']
-------------------------- iter_num = 2 --------------------------
dirpath = 'Python/code/📂folder2'
dirnames = []
filenames = ['exp1.md', 'exp2.txt']
```

### 1.3.3 例子：撰写一个脚本，使其可以遍历一个文件夹下的所有文件，并返回完整的文件路径

想要完成这个例子，如果使用 `os.listdir(path)` 其实是不太现实的，因为 `os.listdir(path)` 只遍历一层文件夹，不会进行深度的遍历。那么想要完成这个脚本，使用 `os.walk(path)` 是非常适合的。

要遍历的文件夹结构如下：

```
Python/code
├── exp1.py
├── exp2.py
├── exp3.py
├── exp4.py
├── exp5.py
├── exp6.py
├── exp7.py
├── 📂folder1
│   ├── exp1.py
│   └── exp2.c
└── 📂folder2
    ├── exp1.md
    └── exp2.txt

2 directories, 11 files
```

脚本代码如下：

```python
import os


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
```

```
all_files_path = ['Python/code/exp1.py', 'Python/code/exp2.py', 'Python/code/exp3.py', 
'Python/code/exp4.py', 'Python/code/exp5.py', 'Python/code/exp6.py', 'Python/code/exp7.py', 
'Python/code/📂folder1/exp1.py', 'Python/code/📂folder1/exp2.c', 'Python/code/📂folder2/exp1.md', 
'Python/code/📂folder2/exp2.txt']

len(all_files_path) = 11
```

我们可以看到，脚本已经 ✅  work 了。

## 1.4 os.path.exists(path)

**作用**：传入一个 path 路径，判断指定路径下的目录是否存在。

+ 存在返回 `True`
+ 不存在返回 `False`

> 💡  这个函数既可以判断文件是否存在，也可以判断📂文件夹是否存在！

### 1.4.1 例子：判断一个文件夹是否存在

```python
import os


flag = os.path.exists('Python/code')
print(f"✅  文件夹存在") if flag else print(f"❌  文件夹不存在！")
```

```
✅  文件夹存在
```

### 1.4.2 例子：检查文件是否存在<a id=1.4.2></a>

```python
import os
import warnings


filepath = 'Python/code/exp100000.py'
if not os.path.exists(filepath):
    warnings.warn(f"⚠️  文件 {filepath} 并不存在！")
else:
    ...
```

> 💡  在平常的工作中，我们自己可以确定文件/📂文件夹存在和不存在的逻辑。

### 1.4.3 例子：创建文件夹

```python
import os


wanna_create_folder_path = 'Python/code'
if not os.path.exists(wanna_create_folder_path):
    os.mkdir(wanna_create_folder_path)
    print(f"✅  文件夹不存在，已创建！")
else:
    print(f"⚠️  文件夹已经存在，无需创建！")
```

```
⚠️  文件夹已经存，无需创建！
```

> ⚠️  OBS：这是一个非常常见的代码。

## 1.5 os.mkdir(path)

**作用**：传入一个 path 路径，创建<font color='red'><b>单层(单个)</b></font>文件夹。

❗  OBS：如果文件夹存在，就会报错。因此创建文件夹之前，一般使用 `os.path.exists(path)` 判断文件夹是否存在（与[1.4.2 例子：检查文件是否存在](#1.4.2)）。

### 1.5.1 例子：创建一个不存在的文件夹

```python
import os


try:
    os.mkdir(f"Python/code/📂folder2")  # 已经存在的文件夹
    print(f"✅  文件夹已创建！")
except Exception as e:
    print(f"{e = }")

try:
    os.mkdir(f"Python/code/📂folder3")  # 不存在的文件夹
    print(f"✅  文件夹已创建！")
except Exception as e:
    print(f"{e = }")
```

```
e = FileExistsError(17, 'File exists')
✅  文件夹已创建！
```

我们可以看到，由于 `Python/code/📂folder2` 这个文件夹是存在的，所以 `os.mkdir(path)` 直接报错了。所以在使用 `os.mkdir(path)` 之前，还是要使用 `os.path.exists(path)` 检查该文件夹是否存在。

### 1.5.2 例子：创建一个父目录也不存在的文件夹 <a id=1.5.2></a>

```python
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
```

```
⚠️  Python/docs 不存在!
⚠️  Python/docs/📂folder1 不存在!
e = FileNotFoundError(2, 'No such file or directory')
```

`os.mkdir` 要求传入的路径的父级文件夹存在，否则会报错 🤣。

## 1.6 os.makedirs(path, exist_ok=False)

**作用**：传入一个 path 路径，生成一个<font color='blue'><b>递归的</b></font>文件夹；

❗  OBS：

1. `exist_ok` 默认为 `False`，即<font color='red'><b>如果文件夹已经存在，就会报错</b></font>。反之，`exist_ok=True`，则文件夹存在也不会报错。因此创建文件夹之前，一般也要使用 `os.path.exists(path)` 函数判断文件夹是否存在，否则开启 `exist_ok` 参数。
2. 与 `makedirs(path, exist_ok=False)` 不同，`os.mkdir()` 并没有 `exist_ok` 参数。


### 1.6.1 例子：四种情况汇总

```python
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
```

```
---------- 使用 os.makedirs(exist_ok=False) 创建已经存在的文件夹 ----------
❌  e = FileExistsError(17, 'File exists')

---------- 使用 os.makedirs(exist_ok=True) 创建已经存在的文件夹 ----------
✅  文件夹 Python/code/📂folder3 已创建！

---------- 使用 os.makedirs(exist_ok=False) 创建已经不存在的文件夹 ----------
✅  文件夹 Python/code/📂folder4 已创建！

---------- 使用 os.makedirs(exist_ok=True) 创建已经不存在的文件夹 ----------
✅  文件夹 Python/code/📂folder5 已创建！
```

### 1.6.1 例子

通过上面的四个例子其实大家可以看到，`os.makedirs()` 似乎与 `os.mkdir()` 没有什么不同，其实这样的理解是错误的。`os.makedirs()` 的出现主要是“递归”创建文件夹，这是什么意思呢？在 [1.5.2 例子：创建一个父目录也不存在的文件夹](#1.5.2) 中展示了，如果要创建的文件夹的上级文件夹（父级文件夹）不存在，那么 `os.mkdir()` 是会报错的。

那么面对这个问题，`os.makedirs()` 出手了，它主要就是解决父级文件夹不存在时，递归地（顺手地）把父级文件夹也给创建了。

> 💡  OBS：这里说的父级文件夹不仅仅是一层哦，多少层都可以。

我们举个例子：

```python
import os


# 创建一个深层的、不存在的文件夹
dirpath = 'Python/code/📂folder6/📂aaa/📂bbb/📂ccc/'

try:
    os.makedirs(dirpath)
    print(f"✅  {dirpath} 创建完毕！")
except Exception as e:
    print(f"❌  {e}")
```

```
✅  Python/code/📂folder6/📂aaa/📂bbb/📂ccc/ 创建完毕！
```

我们可以使用 `tree` 命令查看一下：

```
Python/code/📂folder6
└── 📂aaa
    └── 📂bbb
        └── 📂ccc

3 directories, 0 files
```

可以发现，`os.makedirs()` 可以帮助我们创建父级文件夹不存在的文件夹，适用范围比 `os.mkdir()` 要广泛一些。但是请注意：`os.makedirs()` 因为可以递归创建文件夹，所以在使用之前一定要确认好路径 🤣。

## 1.7 os.rmdir(path)

**作用**：传入一个 path 路径，删除指定路径下的<font color='red'><b>空文件夹</b></font>。

> ⚠️  OBS：该方法只能删除空文件夹，删除非空文件夹会报错。


### 1.7.1 例子：删除一个空文件夹

```python
import os

dirpath = 'Python/code/📂folder6/📂aaa/📂bbb/📂ccc/'
try:
    os.rmdir(dirpath)
    print(f"✅  已成功删除 {dirpath} 文件夹！")
except Exception as e:
    print(f"❌  {e}")
```

```
✅  已成功删除 Python/code/📂folder6/📂aaa/📂bbb/📂ccc/ 文件夹！
```

### 1.7.2 例子：删除一个非空文件夹

非空有两种含义，一种是该文件夹里面有文件，一种是文件夹里面有文件夹。所以我们都测试一下。

```
Python/code/📂folder6
└── 📂aaa
    └── 📂bbb

2 directories, 0 file


Python/code/📂folder7
├── file1
├── file2.py
└── file3.md

0 directories, 3 files
```

```python
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
```

```
---------- 文件夹里面有文件夹 ----------
❌  [Errno 39] Directory not empty: 'Python/code/📂folder6/'

---------- 文件夹里面有文件 ----------
❌  [Errno 39] Directory not empty: 'Python/code/📂folder7/'
```

我们发现我们的猜测是对的：一个文件夹不管里面是有文件夹还是有文件，都不能称之为空文件夹！

### 1.7.3 扩展：删除一个非空的文件夹

在 Python 中，`os.rmdir` 是删除一个空文件夹，那么有没有什么方法可以删除一个非空的文件夹呢？

我们可以使用 `shutil.rmtree` 函数。这个函数会递归地删除指定目录及其所有子目录和文件。

以下是一个使用 `shutil.rmtree` 删除非空文件夹的例子：

```python
import shutil


print(f"---------- 文件夹里面有文件夹 ----------")
dirpath = 'Python/code/📂folder6/'
try:
    shutil.rmtree(dirpath)
    print(f"✅  已成功删除 {dirpath} 文件夹！")
except Exception as e:
    print(f"❌  {e}")
    
    
print(f"\n---------- 文件夹里面有文件 ----------")
dirpath = 'Python/code/📂folder7/'
try:
    shutil.rmtree(dirpath)
    print(f"✅  已成功删除 {dirpath} 文件夹！")
except Exception as e:
    print(f"❌  {e}")
```

```
---------- 文件夹里面有文件夹 ----------
✅  已成功删除 Python/code/📂folder6/ 文件夹！

---------- 文件夹里面有文件 ----------
✅  已成功删除 Python/code/📂folder7/ 文件夹！
```

⚠️  OBS：`shutil.rmtree` 函数是<font color='red'><b>不可逆的</b></font>，一旦文件夹被删除，就无法恢复其中的内容。因此，在执行删除操作之前，请确保我们做了正确的备份或者确认删除是正确的操作！！！


## 1.8 os.path.join(path_1, path_2) <a id=1.8></a>

**作用**：传入<font color='red'><b>多个</b></font> path 路径，将该路径拼接起来，形成一个新的完整路径。

> 💡  OBS：这个函数推荐大家经常用，因为在 Linux 和 Windows 中，系统的拼接字符是不一样的：
> - Windows：`\`
> - Linux: `/`

为了提高我们代码的可用度，建议拼接字符串就使用它！

### 1.8.1 例子：拼接多个字符串


```python
import os


dirname1 = 'Python'
dirname2 = 'code'
dirname3 = '📂folder1'

dirpath = os.path.join(dirname1, dirname2, dirname3)
print(f"{dirpath = }")
```

```
dirpath = 'Python/code/📂folder1'
```

> 💡  OBS：
> 1. 这个函数不仅仅可以拼接两个路径，可以是多个。
> 2. 结果随着操作系统而变化，我使用的系统是 Linux，在 Windows 系统中，结果为：`dirpath = 'Python\code\📂folder1'`

## 1.9 os.path.dirname(path)

**作用**：传入一个完整的文件（文件夹）路径（相对操作系统），只获取其所在文件夹的路径（dirpath）。

### 1.9.1 例子

```
Python/code/📂folder2
├── exp1.md
├── exp2.txt
└── 📂sub_folder

1 directory, 2 files
```

```python
import os


print(f"---------- os.path.dirname(path) 接收的是文件路径 ----------")
filepath = 'Python/code/📂folder2/exp2.txt'
dirpath = os.path.dirname(filepath)
print(f"{dirpath = }")

print(f"\n---------- os.path.dirname(path) 接收的是📂文件夹路径 ----------")
filepath = 'Python/code/📂folder2/📂sub_folder'
dirpath = os.path.dirname(filepath)
print(f"{dirpath = }")
```

```
---------- os.path.dirname(path) 接收的是文件路径 ----------
dirpath = 'Python/code/📂folder2'

---------- os.path.dirname(path) 接收的是📂文件夹路径 ----------
dirpath = 'Python/code/📂folder2'
```

## 1.10 os.path.basename(path)

**作用**：传入一个完整的文件（文件夹）路径，只获取其文件名（filename）。

### 1.10.1 例子：

```
Python/code/📂folder2
├── exp1.md
├── exp2.txt
└── 📂sub_folder

1 directory, 2 files
```

```python
import os


print(f"---------- os.path.basename(path) 接收的是文件路径 ----------")
filepath = 'Python/code/📂folder2/exp2.txt'
dirpath = os.path.basename(filepath)
print(f"{dirpath = }")

print(f"\n---------- os.path.basename(path) 接收的是📂文件夹路径 ----------")
filepath = 'Python/code/📂folder2/📂sub_folder'
dirpath = os.path.basename(filepath)
print(f"{dirpath = }")
```

```
---------- os.path.basename(path) 接收的是文件路径 ----------
dirpath = 'exp2.txt'

---------- os.path.basename(path) 接收的是📂文件夹路径 ----------
dirpath = '📂sub_folder'
```

## 1.11 os.path.split(path)

**作用**：传入一个完整的 path 路径，将其拆分为 dirpath 和 filename。

> 💡  Tips：`os.path.split()` 等价于同时使用了 `os.path.dirname()` 和 `os.path.basename()`。

### 1.11.1 例子

```
Python/code/📂folder2
├── exp1.md
├── exp2.txt
└── 📂sub_folder

1 directory, 2 files
```

```python
import os


print(f"---------- os.path.split(path) 接收的是文件路径 ----------")
filepath = 'Python/code/📂folder2/exp2.txt'
dirpath, filename = os.path.split(filepath)
print(f"{dirpath = }")
print(f"{filename = }")
print(f"{os.path.dirname(filepath) = }")
print(f"{os.path.basename(filepath) = }")

print(f"\n---------- os.path.split(path) 接收的是📂文件夹路径 ----------")
filepath = 'Python/code/📂folder2/📂sub_folder'
dirpath, filename = os.path.split(filepath)
print(f"{dirpath = }")
print(f"{filename = }")
print(f"{os.path.dirname(filepath) = }")
print(f"{os.path.basename(filepath) = }")
```

```
---------- os.path.split(path) 接收的是文件路径 ----------
dirpath = 'Python/code/📂folder2'
filename = 'exp2.txt'
os.path.dirname(filepath) = 'Python/code/📂folder2'
os.path.basename(filepath) = 'exp2.txt'

---------- os.path.split(path) 接收的是📂文件夹路径 ----------
dirpath = 'Python/code/📂folder2'
filename = '📂sub_folder'
os.path.dirname(filepath) = 'Python/code/📂folder2'
os.path.basename(filepath) = '📂sub_folder'
```

## 1.12 os.path.splitext(path)

**作用**：传入一个完整的 path 路径，将路径拆分成两部分：

1. dirpath + filename（不包括扩展名）
2. 文件的扩展名

### 1.12.1 例子

```
Python/code/📂folder2
├── exp1.md
├── exp2.txt
└── 📂sub_folder

1 directory, 2 files
```

```python
import os


print(f"---------- os.path.splitext(path) 接收的是文件路径 ----------")
filepath = 'Python/code/📂folder2/exp2.txt'
prefix, extension = os.path.splitext(filepath)
print(f"{prefix = }")
print(f"{extension = }")

print(f"\n---------- os.path.splitext(path) 接收的是📂文件夹路径 ----------")
filepath = 'Python/code/📂folder2/📂sub_folder'
prefix, extension = os.path.splitext(filepath)
print(f"{prefix = }")
print(f"{extension = }")
```

```
---------- os.path.splitext(path) 接收的是文件路径 ----------
prefix = 'Python/code/📂folder2/exp2'
extension = '.txt'

---------- os.path.splitext(path) 接收的是📂文件夹路径 ----------
prefix = 'Python/code/📂folder2/📂sub_folder'
extension = ''
```

### 1.12.2 例子：搭配 os.path.basename 使用

```python
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
```

```
---------- os.path.splitext(path) 接收的是文件路径 ----------
prefix = 'exp2'
extension = '.txt'

---------- os.path.splitext(path) 接收的是📂文件夹路径 ----------
prefix = '📂sub_folder'
extension = ''
```

## 1.13 os.path.isdir(path)

**作用**：传入一个完整的文件路径，判断它是否是文件夹。

### 1.13.1 例子

```python
import os


print(f"---------- os.path.isdir(path) 接收的是文件路径 ----------")
filepath = 'Python/code/📂folder2/exp2.txt'
flag = os.path.isdir(filepath)
print(f"{flag = }")

print(f"\n---------- os.path.isdir(path) 接收的是📂文件夹路径 ----------")
filepath = 'Python/code/📂folder2/📂sub_folder'
flag = os.path.isdir(filepath)
print(f"{flag = }")

print(f"\n---------- os.path.isdir(path) 接收的是不存在的路径 ----------")
filepath = 'Python/code/📂folder2/XXXXXX'
flag = os.path.isdir(filepath)
print(f"{flag = }")
```

```
---------- os.path.isdir(path) 接收的是文件路径 ----------
flag = False

---------- os.path.isdir(path) 接收的是📂文件夹路径 ----------
flag = True

---------- os.path.isdir(path) 接收的是不存在的路径 ----------
flag = False
```

> ⚠️  OBS：对于不存在的路径，`os.path.isdir()` 返回的也是 `False`。

## 1.14 os.path.isfile(path)

**作用**：传入一个完整的文件路径，判断它是否是文件。

### 1.14.1 例子

```python
import os


dirpath = 'Python/code/📂folder1'
filepath = 'Python/code/📂folder1/exp1.py'

print(f"---------- os.path.isfile() 接收的是📂文件夹 ----------")
print(f"{os.path.isfile(dirpath) = }")

print(f"\n---------- os.path.isfile() 接收的是文件 ----------")
print(f"{os.path.isfile(filepath) = }")

print(f"\n---------- os.path.isfile(path) 接收的是不存在的路径 ----------")
print(f"{os.path.isfile('Python/code/XXXXX') = }")
```

```
---------- os.path.isfile() 接收的是📂文件夹 ----------
os.path.isfile(dirpath) = False

---------- os.path.isfile() 接收的是文件 ----------
os.path.isfile(filepath) = True

---------- os.path.isfile(path) 接收的是不存在的路径 ----------
os.path.isfile('Python/code/XXXXX') = False
```

## 1.15 os.path.sep

**作用**：返回当前操作系统的路径分隔符。

> `sep` = separate

之前我们在 [1.8 os.path.join(path_1, path_2)](#1.8) 中说过，在 Windows 和 Linux 系统中，路径分隔符是不同的，具体为：

+ Windows下为：`\`
+ Linux下为：`/`

### 1.15.1 例子：演示用法

```python
import os
import platform

print(f"---------- 当前运行操作系统为：{platform.system()} ----------")
print(f"{os.path.sep = }")
```

```
---------- 当前运行操作系统为：Linux ----------
os.path.sep = '/'
```

我们再用 Windows 运行同样的代码：

```
---------- 当前运行操作系统为：Windows ----------
os.path.sep = '\\'
```

### 1.15.2 💡  Windows 路径分隔符的特殊说明

需要注意的是，虽然我们说 Windows 的路径分隔符是 `\`，但有时候我们发现，路径用的是 `\` 而非 `\\`，这是因为 `\` 字符在 Windows 路径中具有特殊意义，它代表一个<font color='red'><b>转义字符</b></font>。当我们在路径中遇到两个连续的 `\` 字符时，它们会合并成一个单个的 `\` 字符，这被称为转义。

例如，如果我们有一个路径 `C:\Documents and Settings\User\Desktop`，实际上这个路径是表示 `C:\Documents and Settings\User\Desktop`，因为第一个 `\` 字符是转义的。

这种转义机制在 Windows 中用于路径表示，但在 Python 等编程语言中，我们需要特别注意这个转义字符，因为它可能会导致字符串解析错误。

### 1.15.3 raw string

如果我们传入一个 Windows 路径报错了，不要着急修改路径，我们直接在这个路径字符串前面加个 `r` 就可以了。

这是因为在 Python 中，当字符串前面加上 `r` 前缀时，它被标记为原始字符串（raw string）。原始字符串不会将反斜杠 `\` 视为转义字符，而是将其作为普通字符处理。因此，如果我们在 Windows 路径字符串前面加上 `r` 前缀，它就不会将双反斜杠 `\\` 视为一个转义字符，而是作为两个单独的反斜杠处理。

这样做的好处是，我们可以直接在字符串中使用双反斜杠 `\\`，而不会导致转义错误。这对于编写处理 Windows 路径的代码非常有用，尤其是在编写文件路径或文件操作的代码时。

例如，如果我们有一个 Windows 路径字符串，并且我们在没有 `r` 前缀的情况下尝试使用它，可能会遇到错误，因为我们可能意外地将双反斜杠 `\\` 视为一个转义字符。

## 1.16 os.path.getsize(path)

**作用**：传入一个完整的文件路径，返回该文件的大小（单位是 byte，也就是 B）。

> ⚠️  OBS：
> - b: bit，位。
> - B: byte，字（1B = 8b）

### 1.16.1 例子：获取单个文件的大小

文件 `Python/code/📂folder1/exp3.txt` 中的内容为：

```
This is an example file.
This is an example file.
This is an example file.
```

```python
import os


filepath = 'Python/code/📂folder1/exp3.txt'
print(f"{os.path.getsize(filepath) = } bytes")
```

```
os.path.getsize(filepath) = 74 bytes
```

### 1.16.2 例子：获取空文件夹的大小

```python
import os


dirpath = 'Python/code/📂folder5'
print(f"{os.path.getsize(dirpath) = } bytes")
```

```
os.path.getsize(dirpath) = 4096 bytes
```

> 💡  OBS：文件夹的默认大小为 4096 bytes

### 1.16.3 例子：获取包含文件的文件夹的大小

```
Python/code/📂folder2
├── exp1.md
├── exp2.txt
└── 📂sub_folder
    ├── sub_file1.py
    └── sub_file2

1 directory, 4 file
```

```python
import os


dirpath = 'Python/code/📂folder2'
print(f"{os.path.getsize(dirpath) = } bytes")
```

```
os.path.getsize(dirpath) = 4096 bytes
```

`os.path.getsize(path)` 方法<font color='red'><b>并不会统计文件夹里面文件的大小</b></font>，只会返回输入路径文件/文件夹大小。

### 1.16.4 扩展：以更加易读的方式返回文件大小

```python
import os


def get_file_size(filepath, unit='MB', ndigits=4):
    """获取文件大小
    Args:
        fp (str): 文件路径
        unit (str): 单位选项，可以是'KB', 'MB', 'GB'等
        ndigits (int): 小数点后保留的位数
    Returns:
        float: 文件大小(默认为MB)
    """
    
    # 获取文件大小（字节）
    file_size_bytes = os.path.getsize(filepath)
    unit = unit.upper()
    
    # 单位到字节倍数的映射
    unit_multipliers = {
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024,
    }
    
    # 根据单位转换文件大小
    if unit in unit_multipliers:
        multiplier = unit_multipliers[unit]
        file_size = round(file_size_bytes / multiplier, ndigits=ndigits)
    else:
        # 默认或未知单位时使用MB
        file_size = round(file_size_bytes / (1024 * 1024), ndigits=ndigits)
        unit = 'MB'
    return file_size


if __name__ == '__main__':
    filepath = 'Python/Python中的os模块和sys模块.md'
    filesize = get_file_size(filepath=filepath, unit='MB', ndigits=4)
    print(f"{filesize = } MB")
```

```
filesize = 0.0346 MB
```

# 2. sys 模块

sys 模块是与 Python 解释器交互的一个接口。sys 模块提供了许多函数和变量来处理 Python 运行时环境的不同部分。

## 2.1 sys.argv：查看运行代码时传入的参数

在解释器启动后，argv 列表包含了传递给脚本的所有参数, 列表的第一个元素为脚本自身的名称。

- `sys.argv[0]`：表示程序自身
- `sys.argv[1]`：表示程序的第一个参数
- `sys.argv[2]`：表示程序的第二个参数

### 2.1.1 例子

```python
import sys

print(f"{sys.argv = }")
```

```
sys.argv = ['Python/code/🎁sys/exp1.py']
```

### 2.1.2 例子：给脚本传入参数

```python
import sys

for idx, item in enumerate(sys.argv, start=0):
    print(f"[参数-{idx}] {item} (type={type(item)})")
```

那么怎么给一个脚本传入参数呢？其实非常简单：

```bash
python Python/code/🎁sys/exp2.py a='传入的第一个参数' 2 随便传 d={123} list, Le0v1n, [123, aaa] 🤣 使用空格进行分隔嗷 123
```

结果如下：

```
[参数-0] Python/code/🎁sys/exp2.py (type=<class 'str'>)
[参数-1] a=传入的第一个参数 (type=<class 'str'>)
[参数-2] 2 (type=<class 'str'>)
[参数-3] 随便传 (type=<class 'str'>)
[参数-4] d={123} (type=<class 'str'>)
[参数-5] list, (type=<class 'str'>)
[参数-6] Le0v1n, (type=<class 'str'>)
[参数-7] [123, (type=<class 'str'>)
[参数-8] aaa] (type=<class 'str'>)
[参数-9] 🤣 (type=<class 'str'>)
[参数-10] 使用空格进行分隔嗷 (type=<class 'str'>)
[参数-11] 123 (type=<class 'str'>)
```

⚠️  OBS：

1. <font color='red'><b>传入的参数其实都是字符串</b></font>！
2. 其实传入参数是使用 `空格` 作为分隔的，并非逗号！

## 2.2 sys.exit(n)：手动退出程序

在 Python 中，`sys.exit(n)` 函数用于退出程序。当调用 `sys.exit(n)` 时，它会引发一个 `SystemExit` 异常。如果这个异常没有被捕获，Python 解释器将会退出。参数 `n` 是退出时的状态码，非零值表示异常退出。<font color='blue'>在大多数操作系统中，退出状态码 `0` 表示程序正常退出，而非零的退出状态码通常表示程序异常终止</font>。

在实际应用中，`sys.exit()` 经常用于程序中需要提前退出的情况，比如错误处理或者用户中断等。

### 2.2.1 例子：正常退出 <a id=2.2.1></a>

```python
import sys


print("执行 sys.exit() 之前")
sys.exit()  # 默认为 0，表示正常退出
print("执行 sys.exit() 之后")
```

### 2.2.2 例子：异常退出

```python
import sys


print("执行 sys.exit(7) 之前")
sys.exit(7)  # 默认为 0，表示正常退出
print("执行 sys.exit() 之后")
```

结果我们用图片展示：

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-04-19-15-53-05.png
    width=100%>
    <center></center>
</div></br>

我们可以发现，在 [2.2.1 例子：正常退出](#2.2.1) 中（也就是 `exp3.py` 文件），程序执行了 `sys.exit()` 之后，因为状态默认为 `0`，所以程序是正常退出的，从图片中也可以看到，终端对应位置是一个 <font color='blue'>●</font>。但是当状态不是 `0` 时，那么程序虽然退出了，但终端对应位置是一个 <font color='red'>×</font>，这意味着程序并非正常退出。

> 💡  Tips：`sys.exit()` 等价于 `exit()`。我们可以使用 `exit()` 来简化 `sys.exit()`。

## 2.3 sys.version：获取 Python 解释器版本

```python
import sys


print(f"当前激活的环境中，Python 版本为：{sys.version}")
```

```
当前激活的环境中，Python 版本为：3.8.18 (default, Sep 11 2023, 13:40:15) 
[GCC 11.2.0]
```

## 2.4 sys.platform：返回操作系统的平台名称

```python
import sys


print(f"当前操作系统为：{sys.platform}")
```

```
当前操作系统为：linux
```

## 2.5 sys.stdin.readline() 和 input()：等待用户输入

`sys.stdin.readline()` 相当于`input()`，二者的区别在于：

1. `input()` 不会读入 `\n`，`sys.stdin.readline()` 会读入 `\n`
2. `input()` 默认参数为字符串，可充当提示语，而 `sys.stdin.readline` 没有这个参数

> 在日常工作中，我们还是习惯使用 `input()` 而非 `sys.stdin.readline()`
>
> stdin: standard input, 标准输入

### 2.5.1 例子：sys.stdin.readline()

```
import sys

input_content = sys.stdin.readline()
print(f"{input_content = }")
```

终端输入：

```bash
python Python/code/🎁sys/exp7.py
Hello World!
```

结果：

```
input_content = 'Hello World!\n'
```

### 2.5.2 例子：input()

```python
import sys


input_content = input(f"请输入一行文字：")
print(f"{input_content = }")
```

终端输入：

```bash
python Python/code/🎁sys/exp8.py
请输入一行文字：Hello World!!!
```

结果：

```
input_content = 'Hello World!!!'
```

