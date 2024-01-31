# 1. 被导入的包和运行的文件在同一目录下

比如我们有下面这样的一个目录结构：

```
./
├── A
│   ├── AA
├── B
│   └── BB
├── Python
│   ├── resolve_import_methods
│       ├── test.py
│       └── utils.py
```

其中 `Python/resolve_import_methods/utils.py` 代码中的内容如下：

```python
class_dict = {
    0: 'cat',
    1: 'dog'
}


def example_fn(a, b):
    return a + b
```

`Python/resolve_import_methods/test.py` 代码中的内容如下：

```python
from Python.utils import class_dict, example_fn


if __name__ == "__main__":
    print(f"Code starts!")
```

这样写了之后，VSCode 会报错：

<div align=center>
    <img src=./imgs_markdown/2024-01-31-15-07-08.png
    width=70%>
    <center></center>
</div>

但是在 `import` 时又会自动提示 `class_dict` 和 `example_fn` 这两个变量，那就有点奇怪了。所以我们直接运行看看：

```
Traceback (most recent call last):
  File "Python/resolve_import_methods/test.py", line 1, in <module>
    from Python.utils import class_dict, example_fn
ModuleNotFoundError: No module named 'Python'
```

编译期告诉我们没有名为 `'Python'` 的模块。那我们直接去掉 `'Python'` 行不行，就当是在当前目录下运行的：

<div align=center>
    <img src=./imgs_markdown/2024-01-31-15-10-26.png
    width=55%>
    <center></center>
</div>

当我们去掉 `'Python'` 之后，VSCode 不报错了，那我们运行试试：

```
Code starts!
```

程序正常运行了。这说明一个问题：在运行当前 `.py` 文件后，该文件中的路径就是以当前 `.py` 文件为工作目录。

当我们使用 `from Python.utils import xxx` 时，我们以为终端的目录即为当前的工作目录，但上面的实例告诉我们并不是这样的。

# 2. 被导入的包在运行的文件的上一级目录

但这样也会带来一个问题：如果我们的目录结构变成下面这样：

```
./
├── A
│   ├── AA
├── B
│   └── BB
├── Python
│   ├── resolve_import_methods
│   │   ├── test.py
│   └── utils.py
```

即 `utils.py` 不在 `test.py` 的文件夹内，在它的上一级，那么我们的代码还能用吗？

```python
from utils import class_dict, example_fn


if __name__ == "__main__":
    print(f"Code starts!")
```

```
Traceback (most recent call last):
  File "Python/resolve_import_methods/test.py", line 1, in <module>
    from utils import class_dict, example_fn
ModuleNotFoundError: No module named 'utils'
```

与我们的预期一样，并不能正常导入了，因为 `utils.py` 文件并不在 `test.py` 的当前目录下，那我们还能导入吗？我们试试 `.`：

```python
from ..utils import class_dict, example_fn


if __name__ == "__main__":
    print(f"Code starts!")
```

```
Traceback (most recent call last):
  File "Python/resolve_import_methods/test.py", line 1, in <module>
    from ..utils import class_dict, example_fn
ImportError: attempted relative import with no known parent package
```

说明这样的相对引用不太行。那我们能否使用绝对引用？

```python
from Python.utils import class_dict, example_fn


if __name__ == "__main__":
    print(f"Code starts!")
```

```
Traceback (most recent call last):
  File "Python/resolve_import_methods/test.py", line 1, in <module>
    from Python.utils import class_dict, example_fn
ModuleNotFoundError: No module named 'Python'
```

依然不可以。那么为什么呢？

我们可以看一下当前的工作目录：

```python
# from Python.utils import class_dict, example_fn
import os


if __name__ == "__main__":
    print(f"Code starts!")
    print(f"{os.getcwd() = }")
```

```
os.getcwd() = '/mnt/f/Projects/本地代码/Learning-Notebook-Codes'
```

说明代码的工作目录为：`'/mnt/f/Projects/本地代码/Learning-Notebook-Codes'`，那按理来说我们是可以使用绝对引用的，为什么不可以呢？

这里有一个坑：在终端使用命令行的形式运行 `.py` 文件和直接在右上角按 <kbd>运行</kbd> 时的工作路径是不一样的。

<kbd><b>Q</b>uestion</kbd>：但为什么我在终端运行 `.py` 文件还是会提示没有该 module 呢？
<kbd><b>A</b>nswer</kbd>：不知道。目前的解决方案是添加下面的语句：

```python
import sys
sys.path.append('/mnt/f/Projects/本地代码/Learning-Notebook-Codes')
from Python.utils import class_dict, example_fn


if __name__ == "__main__":
    print(f"Code starts!")
    print(f"{os.getcwd() = }")
```

将根目录设置为工作目录，这样就可以随便使用绝对引用了。

<font color='green'>PS：如果有大佬知道怎么搞，麻烦告诉我一下 :smile:</font>


