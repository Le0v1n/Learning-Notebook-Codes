<div align=center>
    <img src=https://img-blog.csdnimg.cn/d0601a5148044167a7fb3394bc91b95b.png
    width=50%>
</div>

<kbd>官方仓库</kbd>：[https://github.com/Textualize/rich](https://github.com/Textualize/rich)

# 1. rich 官方介绍

Rich 是一个 Python 库，可以为您在终端中提供富文本和精美格式。

[Rich 的 API](https://rich.readthedocs.io/en/latest/) 让在终端输出颜色和样式变得很简单。此外，Rich 还可以绘制漂亮的表格、进度条、markdown、语法高亮的源代码以及栈回溯信息（tracebacks）等——开箱即用。

<div align=center>
    <img src=https://img-blog.csdnimg.cn/833133dd808a4c12b6c199bab6419bd8.png
    width=100%>
</div>

## 1.1 安装

使用 `pip` 或其他 PyPI 软件包管理器进行安装。

```sh
python -m pip install rich
```

## 1.2 Rich 的打印功能

想毫不费力地将 Rich 的输出功能添加到您的应用程序中，您只需导入 [rich print](https://rich.readthedocs.io/en/latest/introduction.html#quick-start) 方法，它和 Python 内置的同名函数有着完全一致的函数签名。试试看：

```python
from rich import print

print("Hello, [bold magenta]World[/bold magenta]!", ":vampire:", locals())
```

<div align=center>
    <img src=https://img-blog.csdnimg.cn/f8a2c3ca12194bbaa0069701c8b5fe49.png
    width=90%>
</div>

## 1.3 在交互式命令行（REPL）中使用 Rich

Rich 可以被安装到 Python 交互式命令行中，那样做以后，任何数据结构都可以被漂亮的打印出来，自带语法高亮。

```python
>>> from rich import pretty
>>> pretty.install()
```

<div align=center>
    <img src=https://img-blog.csdnimg.cn/72b5b646d8954e02b1a323faaad691ab.png
    width=90%>
</div>

## 1.4 使用控制台

想要对 Rich 终端内容进行更多控制，请您导入并构造一个[控制台](https://rich.readthedocs.io/en/latest/reference/console.html#rich.console.Console)对象。

```python
from rich.console import Console

console = Console()
```

Console 对象包含一个`print`方法，它和语言内置的`print`函数有着相似的接口。下面是一段使用样例：

```python
console.print("Hello", "World!")
```

您可能已经料到，这时终端上会显示“ Hello World！”。请注意，与内置的“print”函数不同，Rich 会将文字自动换行以适合终端宽度。

有好几种方法可以为输出添加颜色和样式。您可以通过添加`style`关键字参数来为整个输出设置样式。例子如下：

```python
console.print("Hello", "World!", style="bold red")
```

输出如下图：

<div align=center>
    <img src=https://img-blog.csdnimg.cn/d571e567a5b6490995722b4540cc3f30.png
    width=90%>
</div>


这个范例一次只设置了一行文字的样式。如果想获得更细腻更复杂的样式，Rich 可以渲染一个特殊的标记，其语法类似于[bbcode](https://en.wikipedia.org/wiki/BBCode)。示例如下：

```python
console.print("Where there is a [bold cyan]Will[/bold cyan] there [u]is[/u] a [i]way[/i].")
```

<div align=center>
    <img src=https://img-blog.csdnimg.cn/1b8482cb2045438fa6f7826d45c42fa8.png
    width=90%>
</div>

使用`Console`对象，你可以花最少的工夫生成复杂的输出。更详细的内容可查阅 [Console API](https://rich.readthedocs.io/en/latest/console.html) 文档。

## 1.5 Rich Inspect

Rich 提供一个 [inspect](https://rich.readthedocs.io/en/latest/reference/init.html?highlight=inspect#rich.inspect) 函数来给任意的 Python 对象打印报告，比如类（class）、实例（instance）和内置对象（builtin）等。

```python
from rich import inspect


my_list = ["foo", "bar"]
inspect(my_list, methods=True)
```

<div align=center>
    <img src=https://img-blog.csdnimg.cn/dfd41937d74f4a5e987659d7511f8b3a.png
    width=100%>
</div>

查看  [inspect 文档](https://rich.readthedocs.io/en/latest/reference/init.html#rich.inspect)详细了解。

# 2. Rich 库内容

Rich 包含了一系列内置的 _可渲染类型(renderables)_ ，你可以用它们为命令行程序构建出优雅的输出，也可以拿它们来辅助调试你的代码。

## 2.1 日志（Log）

Console 对象有一个与 `print()` 类似的 `log()` 方法，但它会多输出一列内容，里面包含当前时间以及调用方法的文件行号。默认情况下，Rich 将针对 Python 结构和 repr 字符串添加语法高亮。如果您记录一个集合（如字典或列表），Rich 会把它漂亮地打印出来，使其切合可用空间。下面是其中一些功能的示例：

```python
from rich.console import Console
console = Console()

test_data = [
    {"jsonrpc": "2.0", "method": "sum", "params": [None, 1, 2, 4, False, True], "id": "1",},
    {"jsonrpc": "2.0", "method": "notify_hello", "params": [7]},
    {"jsonrpc": "2.0", "method": "subtract", "params": [42, 23], "id": "2"},
]

def test_log():
    enabled = False
    context = {
        "foo": "bar",
    }
    movies = ["Deadpool", "Rise of the Skywalker"]
    console.log("Hello from", console, "!")
    console.log(test_data, log_locals=True)


test_log()
```

以上范例的输出如下：

<div align=center>
    <img src=https://img-blog.csdnimg.cn/34a7c639dd93446c87f9e9558a6593b2.png
    width=100%>
</div>


注意其中的`log_locals`参数会输出一个表格，该表格包含调用 log 方法的局部变量。

log 方法既可用于将常驻进程（例如服务器进程）的日志打印到终端，在调试时也是个好帮手。

## 2.2 日志处理器（Logging Handler）

您还可以使用内置的[处理器类](https://rich.readthedocs.io/en/latest/logging.html)来对 Python 的 logging 模块的输出进行格式化和着色。

## 2.3 Emoji 表情

将名称放在两个冒号之间即可在控制台输出中插入 emoji 表情符。示例如下：

```python
>>> console.print(":smiley: :vampire: :pile_of_poo: :thumbs_up: :raccoon:")
😃 🧛 💩 👍 🦝
```

请谨慎地使用此功能。

## 2.4 表格（Tables）

Rich 可以使用 Unicode 框字符来呈现多变的[表格](https://rich.readthedocs.io/en/latest/tables.html)。Rich 包含多种边框，样式，单元格对齐等格式设置的选项。下面是一个简单的示例：

```python
from rich.console import Console
from rich.table import Column, Table

console = Console()

table = Table(show_header=True, header_style="bold magenta")
table.add_column("Date", style="dim", width=12)
table.add_column("Title")
table.add_column("Production Budget", justify="right")
table.add_column("Box Office", justify="right")
table.add_row(
    "Dec 20, 2019", "Star Wars: The Rise of Skywalker", "$275,000,000", "$375,126,118"
)
table.add_row(
    "May 25, 2018",
    "[red]Solo[/red]: A Star Wars Story",
    "$275,000,000",
    "$393,151,347",
)
table.add_row(
    "Dec 15, 2017",
    "Star Wars Ep. VIII: The Last Jedi",
    "$262,000,000",
    "[bold]$1,332,539,889[/bold]",
)

console.print(table)
```

该示例的输出如下：

<div align=center>
    <img src=https://img-blog.csdnimg.cn/8a7590b159e24ccb9040e5f83a0b309d.png
    width=80%>
</div>

请注意，控制台标记的呈现方式与`print()`和`log()`相同。实际上，由 Rich 渲染的任何内容都可以添加到标题/行（甚至其他表格）中。

`Table`类很聪明，可以调整列的大小以适合终端的可用宽度，并能根据需要对文字折行。

## 2.5 进度条（Progress Bars）

### 2.5.1 示例

Rich 可以渲染多种“无闪烁”的[进度](https://rich.readthedocs.io/en/latest/progress.html)条图形，以跟踪长时间运行的任务。

基本用法：用`track`函数调用任何程序并迭代结果。下面是一个例子：

```python
from rich.progress import track
import time


for step in track(range(100)):
    time.sleep(0.1)
```

添加多个进度条并不难。以下是从文档中获取的示例：

<div align=center>
    <img src=https://img-blog.csdnimg.cn/7a70489453ab4df996b3c75dda4afcdd.gif
    width=80%>
</div>

这些列可以配置为显示您所需的任何详细信息。内置列包括完成百分比，文件大小，文件速度和剩余时间。

要自己尝试一下，请参阅[examples/downloader.py](https://github.com/textualize/rich/blob/master/examples/downloader.py)，它可以在显示进度的同时下载多个 URL。

# 文档
1. [https://rich.readthedocs.io/en/latest/progress.html](https://rich.readthedocs.io/en/latest/progress.html)