```
这是一行测试语言，没有其他的含义.这是一行测试语言, 没有其他的含义.这是一行测试语言, 没有其他的含义。

这是一行测试语言, 没有其他的含义.这是一行测试语言, 没有其他的含义.这是一行测试语言，没有其他的含义。这是一行测试语言, 没有其他的含义.这是一行测试语言，没有其他的含义。
```

# 1. Shift

Shift, 又名换挡键, 一般是 <kbd>Shift + 其他按键</kbd>。

## 1.1 文字多选

我们可以通过 <kbd>Shift + ←/→</kbd> 来进行文字的多选，也可以通过 <kbd>Shift + ↑/↓</kbd> 直接选取多行，演示如 [Fig.1](#Fig.1) 所示。

我们使用 Windows 自带的 notepad.exe 进行演示：

<a id=Fig.1></a>
<div align=center>
    <img src=./imgs_markdown/gif.1.gif
    width=80%>
    <center>Fig.1 Shift + ↑/↓/←/→ 实现文字多选的示例</center>
</div></br>

---

除了使用 <kbd>Shift + ↑↓←→</kbd> 来实现文字多选外，也可以通过 <kbd>Shift + 鼠标左键</kbd> 来实现文字的多选，演示如 [Fig.2](#Fig.2) 所示。

<a id=Fig.2></a>
<div align=center>
    <img src=./imgs_markdown/gif.2.gif
    width=80%>
    <center>Fig.2 Shift + 鼠标左键 实现文字多选的示例</center>
</div></br>

## 1.2 文件多选

同样的, <kbd>Shift</kbd> 除了可以多选文字外，也可以多选文件，这也是经常使用的功能（可以告别 <kbd>Ctrl</kbd> 一个一个选择的痛苦了），演示如 [Fig.3](#Fig.3) 所示。

<a id=Fig.3></a>
<div align=center>
    <img src=./imgs_markdown/gif.3.gif
    width=80%>
    <center>Fig.3 Shift + 鼠标左键实现文件多选的示例</center>
</div></br>

---

有些同学可能想到了，除了使用 <kbd>Shift + 鼠标左键</kbd> 外，还可以使用 <kbd>Shift + ↑/↓</kbd> 进行文件多选，演示如 [Fig.4](#Fig.4) 所示。

<a id=Fig.4></a>
<div align=center>
    <img src=./imgs_markdown/gif.4.gif
    width=80%>
    <center>Fig.4 Shift + ↑/↓ 实现文件多选的示例</center>
</div></br>

## 1.3 左右滑动

正常来说，使用鼠标的滑轮是上下滑动，我们是否想过，可以左右滑动吗？其实是可以的，只需要按住 <kbd>Shift + 滑轮</kbd>，即可实现左右滑动，演示如 [Fig.5](#Fig.5) 所示。

<a id=Fig.5></a>
<div align=center>
    <img src=./imgs_markdown/gif.5.gif
    width=80%>
    <center>Fig.5 Shift + 鼠标滑轮实现左右滑动的示例</center>
</div></br>

# 2. Delete

## 2.1 向右删除一个字符

我们知道，<kbd>←(backspace)</kbd> 是向左一个字符, 这是我们经常使用的一个按键，但是想过一个问题没有，如果我们想向右删除一个字符，应该如何实现呢？<kbd>Delete</kbd> 按键就实现了这个功能。演示如 [Fig.6](#Fig.6) 所示。

<a id=Fig.6></a>
<div align=center>
    <img src=./imgs_markdown/gif.6.gif
    width=80%>
    <center>Fig.6 使用 Delete 实现向右删除一个字符的示例</center>
</div></br>

> 图中闪烁一次的是 <kbd>ctrl + z</kbd>，可以实现“撤回”，即上一次的操作不算数。

## 2.2 删除文件

正常来说，想要删除一个文件，有下面几种方法：

1. 用鼠标右键点击一个文件，然后按“删除”。
2. 用鼠标左键拖动一个文件到“回收站”。

那么除了上述这两种方法外，还有更加优雅的方法吗？答案是肯定的，我们可以使用 <kbd>Delete</kbd> 删除文件，即：使用鼠标左键选中一个（一批）文件，然后按下 <kbd>Delete</kbd> 键即可。效果和上面的两种方法是一样的，文件都会被放入“回收站”中，演示如 [Fig.7](#Fig.7) 所示。

<a id=Fig.7></a>
<div align=center>
    <img src=./imgs_markdown/gif.7.gif
    width=80%>
    <center>Fig.7 使用 Delete 删除文件的示例</center>
</div></br>

我们可以发现，[Fig.7](#Fig.7) 中的示例是将文件放入回收站中，那我们有没有办法让文件不经过回收站，直接被<font color='red'><b>彻底删除</b></font>呢？是有这种方法的，需要借助我们前面提到的 <kbd>Shift</kbd> 方法，即 <kbd>Shift + Delete</kbd> 即可将文件彻底删除，不会进入回收站。

> ⚠️  注意：<font color='red'><b>此过程不可逆</b></font>，请谨慎操作！

具体示例如 [Fig.8](#Fig.8) 所示。

<a id=Fig.8></a>
<div align=center>
    <img src=./imgs_markdown/gif.8.gif
    width=80%>
    <center>Fig.8 使用 Shift + Delete 彻底删除文件的示例的示例</center>
</div></br>

# 3. Home

这个按键各位同学可能非常生疏，可能都不知道它是干嘛用的，这里我介绍几个非常好用的方法，也是我经常会使用的。

## 3.1 光标回到一行文字的开头

当我们在编辑文字的时候，可能突然想回到一行文字的开头添加一些内容，通常的方法是使用鼠标左键点击开头的位置，将光标转移到开头。这样的确是可行的，但现在有一种更加优雅的方法，即使用 <kbd>Home</kbd>，具体示例如 [Fig.9](#Fig.9) 所示。

<a id=Fig.9></a>
<div align=center>
    <img src=./imgs_markdown/gif.9.gif
    width=80%>
    <center>Fig.9 使用 Home 回到一行文字开头的示例</center>
</div></br>

> 💡  需要说明的是，在不同的软件中，<kbd>Home</kbd> 的功能可能不同，比如在 notepad、VSCode 中，<kbd>Home</kbd> 的功能是回到一行文字的开头，而在 Terminal 中，它是回到一段文字的开头。

## 3.2 返回顶部（回到开始的位置）

在浏览网页的时候，我们会发现有一些网站会有一个名为 <kbd>Top</kbd> 的按钮（有的也叫 “**返回顶部**”），按了这个按钮之后，网页会自动返回到开头的位置。但是有一些网站它是没有这个按钮的，那么我们如何回到网页的开头位置呢？我们也可以借助 <kbd>Home</kbd> 按键，但是需要 <kbd>Ctrl + Home</kbd>，具体示例如 [Fig.10](#Fig.10) 所示。

<a id=Fig.10></a>
<div align=center>
    <img src=./imgs_markdown/gif.10.gif
    width=80%>
    <center>Fig.10 使用 Ctrl + Home 回到开头位置的示例</center>
</div></br>

💡  返回顶部不仅仅可以用于网页，基本上什么软件都支持的：

- 在 Word 中，<kbd>Ctrl + Home</kbd> 回到文档的开头。
- 在 VSCode 中，<kbd>Ctrl + Home</kbd> 回到文档的开头。
- 在 Notepad 中，<kbd>Ctrl + Home</kbd> 回到文档的开头。
- ...

## 3.3 选择左边的文字

我们有时候需要选择左边的文字，比如对于 `https://github.com/Le0v1n/Learning-Notebook-Codes` 这个网址，我需要选择 `https://github.com/`，那么我们怎么实现？

我们当然可以长按鼠标左键进行选择，但是这样有时候比较麻烦（我对于需要使用鼠标的操作都不太感冒 🤪）。那我们就可以使用 <kbd>Shift + Home</kbd> 来进行了，具体示例如 [Fig.11](#Fig.11) 所示。

<a id=Fig.11></a>
<div align=center>
    <img src=./imgs_markdown/gif.11.gif
    width=80%>
    <center>Fig.11 使用 Shift + Home 选择左边的文字的示例</center>
</div></br>

## 3.4 选择上面的文件

<kbd>Shift + Home</kbd> 除了可以选择左边的文件外，还可以选择上面的文件，具体示例如 [Fig.12](#Fig.12) 所示。

<a id=Fig.12></a>
<div align=center>
    <img src=./imgs_markdown/gif.12.gif
    width=80%>
    <center>Fig.12 使用 Shift + Home 选择上面的文件的示例</center>
</div></br>

# 4. End

💡  <kbd>End</kbd> 和 <kbd>Home</kbd> 功能类似, 只不过 <font color='red'><b>方向是反着的</b></font>。

## 4.1 光标回到一行文字的结尾

<a id=Fig.13></a>
<div align=center>
    <img src=./imgs_markdown/gif.13.gif
    width=80%>
    <center>Fig.13 使用 End 回到一行文字结尾的示例</center>
</div></br>

> 💡  需要说明的是，在不同的软件中，<kbd>End</kbd> 的功能可能不同，比如在 notepad、VSCode 中，<kbd>End</kbd> 的功能是回到一行文字的结尾，而在 Terminal 中，它是回到一段文字的结尾。

## 4.2 返回底部（回到末尾的位置）

<a id=Fig.14></a>
<div align=center>
    <img src=./imgs_markdown/gif.14.gif
    width=80%>
    <center>Fig.14 使用 Ctrl + End 回到结尾位置的示例</center>
</div></br>

💡  返回底部不仅仅可以用于网页，基本上什么软件都支持的：

- 在 Word 中，<kbd>Ctrl + Home</kbd> 回到文档的末尾。
- 在 VSCode 中，<kbd>Ctrl + Home</kbd> 回到文档的末尾。
- 在 Notepad 中，<kbd>Ctrl + Home</kbd> 回到文档的末尾。
- ...

## 4.3 选择右边的文字

<a id=Fig.15></a>
<div align=center>
    <img src=./imgs_markdown/gif.15.gif
    width=80%>
    <center>Fig.15 使用 Shift + End 选择右边的文字的示例</center>
</div></br>

## 4.4 选择下面的文件

<a id=Fig.16></a>
<div align=center>
    <img src=./imgs_markdown/gif.16.gif
    width=80%>
    <center>Fig.16 使用 Shift + End 选择下面的文件的示例</center>
</div></br>

# 5. Insert

<kbd>insert</kbd> 是一个不常用的按键，它主要有两个作用，我们以此介绍一下。

## 5.1 插入/改写

正常来说，我们在光标处输入文字，不管光标后面有没有其他文字，都是会正常输入的，这个叫做“插入”模式，如 [Fig.17](#Fig.17) 所示。

<a id=Fig.17></a>
<div align=center>
    <img src=./imgs_markdown/gif.17.gif
    width=80%>
    <center>Fig.17 正常输入（“插入”模式）的示例</center>
</div></br>

此时，我们正处于“插入”模式，在 Word 中，如 [Fig.18](#Fig.18) 所示。

<a id=Fig.18></a>
<div align=center>
    <img src=./imgs_markdown/2024-04-04-00-16-54.png
    width=80%>
    <center></center>
    <center>Fig.18 “插入”模式示意图</center>
</div></br>

当我们在 Word 中不小心按到 <kbd>Insert</kbd> 按键后，“插入”模式会变为“改写”模式。此时如果光标后面有其他文字，那我们输入的时候，其他文字会被覆盖，如 [Fig.19](#Fig.19) 所示。

<a id=Fig.19></a>
<div align=center>
    <img src=./imgs_markdown/gif.19.gif
    width=80%>
    <center>Fig.19 “改写”模式的示例</center>
</div></br>

我们可以发现，下面的状态从原来的“插入”变为了“改写”。

💡  一般来说，如果在某个文字编辑软件中开启了“改写”模式，那么一般会有一个标识（多数情况下，少数情况没有标识提醒），如下图所示：

<div align=center>
    <img src=https://img-blog.csdnimg.cn/20210602162938481.png
    width=80%>
</div></br>
<div align=center>
    <img src=https://img-blog.csdnimg.cn/20210602162948122.png
    width=86%>
</div></br>

## 5.2 Shift + Insert

众所周知，复制是 <kbd>Ctrl + C</kbd>，粘贴是 <kbd>Ctrl + V</kbd>，那么其实使用 <kbd>Shift + Insert</kbd> 也可以粘贴，如 [Fig.20](#Fig.20) 所示。

<a id=Fig.20></a>
<div align=center>
    <img src=./imgs_markdown/gif.20.gif
    width=80%>
    <center>Fig.20 Shift + Insert 实现粘贴的示例</center>
</div></br>

> 💡  <kbd>Shift + Insert</kbd> 一般是用在 Terminal 中，我们平常用 <kbd>Ctrl + V</kbd> 即可。

# 6. PageUp 和 PageDown

在浏览网页、观看文档的时候，我们如果想往下滑动，那么常见的操作有：

1. 使用鼠标滚轮上下滑动
2. 使用鼠标中间

具体操作如 [Fig.21](#Fig.21) 所示。

<a id=Fig.21></a>
<div align=center>
    <img src=./imgs_markdown/gif.21.gif
    width=80%>
    <center>Fig.21 普通的浏览方式</center>
</div></br>

> 💡  按一下鼠标中间的操作是有效的，只不过没有录制出来而已，同学们可以自己实操一下。

那么除了这两种方式外，还有其他优雅的方式吗？有些同学可能会说，可以使用 <kbd>↑/↓</kbd> 来进行浏览，但是这样太慢了，而且也不跟手。

目前一种比较优雅的方式是使用 <kbd>PageUp 和 PageDown</kbd>，它们的功能如下所示：

- <kbd>PageUp</kbd>：往上翻<font color='red'><b>一页</b></font>
- <kbd>PageDown</kbd>：往下翻<font color='red'><b>一页</b></font>

具体操作如 [Fig.22](#Fig.22) 所示。

<a id=Fig.22></a>
<div align=center>
    <img src=./imgs_markdown/gif.22.gif
    width=80%>
    <center>Fig.22 使用 PageUp 和 PageDown 的浏览方式</center>
</div></br>

> 💡  <kbd>PageUp 和 PageDown</kbd> 也是根据不同的软件有着不同的动作，大部分场景下都是翻一页，而有些场景下是翻半页。不管怎么说，都是用来翻页的。

# 7. Ctrl

## 7.1 全选

众所周知，<kbd>Ctrl + A</kbd> 可以实现全选，这个功能不仅可以用于文字编辑（如 [Fig.23](#Fig.23) 所示），也可以用于文件选择（如 [Fig.24](#Fig.24) 所示）。

<a id=Fig.23></a>
<div align=center>
    <img src=./imgs_markdown/gif.23.gif
    width=80%>
    <center>Fig.23 文字编辑中的全选示例</center>
</div></br>

<a id=Fig.24></a>
<div align=center>
    <img src=./imgs_markdown/gif.24.gif
    width=80%>
    <center>Fig.24 文件选择中的全选示例</center>
</div></br>

## 7.2 复制/剪切/粘贴

这没什么好说的，其功能如下：

- 复制：<kbd>Ctrl + C</kbd>
- 剪切：<kbd>Ctrl + X</kbd>
- 粘贴：<kbd>Ctrl + V</kbd>
- 撤回：<kbd>Ctrl + Z</kbd>

> 💡  上述四种功能也是即可以用于文字编辑，也可以用于文件。

## 7.3 文字加粗

在 Word、Markdown 等文本编辑器中，一般使用 <kbd>Ctrl + B</kbd> 来实现文本加粗，如 [Fig.25](#Fig.25) 所示。

<a id=Fig.25></a>
<div align=center>
    <img src=./imgs_markdown/gif.25.gif
    width=80%>
    <center>Fig.25 在 Word 中加粗文字的示例</center>
</div></br>

> 💡  这里的加粗其实就是将字体从原本的 “regular” 变为了 “bold”。

## 7.4 文字变为斜体

同样的，在 Word、Markdown 等文本编辑器中，一般使用 <kbd>Ctrl + I</kbd> 来将文本变为斜体，如 [Fig.26](#Fig.26) 所示。

<a id=Fig.26></a>
<div align=center>
    <img src=./imgs_markdown/gif.26.gif
    width=80%>
    <center>Fig.26 在 Word 中将文本变为斜体的示例</center>
</div></br>

## 7.5 改变字体大小

在 Word 中，可以使用 <kbd>Ctrl + [/]</kbd> 来减小/增加字体大小，功能如下：

- <kbd>Ctrl + [</kbd>：减小字体
- <kbd>Ctrl + ]</kbd>：增大字体

演示如 [Fig.27](#Fig.27) 所示。

<a id=Fig.27></a>
<div align=center>
    <img src=./imgs_markdown/gif.27.gif
    width=80%>
    <center>Fig.27 在 Word 中，Ctrl + [/] 改变字体大小的示例</center>
</div></br>

在 VSCode 中可以通过 <kbd>Ctrl + 加号/减号</kbd> 来改变字体大小（其实是改变了缩放的倍率，不单单是改变字体大小）。

> 💡  在不同的软件中，<kbd>Ctrl + [/]</kbd> 有着不同的效果。

## 7.6 按词组移动光标

我们一般在编辑文字的时候会经常使用 <kbd>←→</kbd> 来移动光标的位置，但这样移动光标有一个问题：太慢了，每次只能移动一个字。那么有没有办法实现一次移动一个词呢？答案是肯定的，我们可以使用 <kbd>Ctrl + ←→</kbd> 来实现一次跳过一个词，演示如 [Fig.28](#Fig.28) 所示。

<a id=Fig.28></a>
<div align=center>
    <img src=./imgs_markdown/gif.28.gif
    width=80%>
    <center>Fig.28 Ctrl + ←/→ 按词组移动光标的示例（中文）</center>
</div></br>

我们可以看到，使用 <kbd>Ctrl + ←→</kbd> 移动光标的时候，光标会按着词组进行移动。这个技巧不光可以用于中文，也可以用于英文，英文的演示如 [Fig.29](#Fig.29) 所示。

<a id=Fig.29></a>
<div align=center>
    <img src=./imgs_markdown/gif.29.gif
    width=80%>
    <center>Fig.29 Ctrl + ←/→ 按词组移动光标的示例（英文）</center>
</div></br>

> 💡  需要特别说明的是：在一般的软件中，如 Word、Notepad 这些微软系软件中，因为都是支持中文的（可以识别中文组成的词语），所以可以使用 <kbd>Ctrl + ←→</kbd> 进行按词组移动光标。但对于一些不支持分词的软件，如 VSCode，我们可以安装插件 [CJK Word Handler](https://github.com/sharzyl/cjk-word-handler.git) 来实现 <kbd>Ctrl + ←→</kbd> 按词组移动光标。对于一些没有插件的软件而言，<kbd>Ctrl + ←→</kbd> 就无法正确地识别中文词汇，光标移动的时候可能不那么智能。

## 7.7 按词组删除文字

我们既然可以按词组移动光标，那么也是可以按词组删除文字的，即将原本的 <kbd>Backspace</kbd> 和 <kbd>Delete</kbd> 与 <kbd>Ctrl</kbd> 进行组合即可，其功能如下：

- <kbd>Ctrl + Backspace</kbd>：向左删除一个单词
- <kbd>Ctrl + Delete</kbd>：向右删除一个单词

具体演示如 [Fig.30](#Fig.30) 所示。

<a id=Fig.30></a>
<div align=center>
    <img src=./imgs_markdown/gif.30.gif
    width=80%>
    <center>Fig.30 Ctrl + Backspace/Delete 按词组删除文字</center>
</div></br>

# 8. Alt

<kbd>Alt</kbd> 是 alter 转换的意思，这个按键我们可能使用的不多，下面我为同学们介绍几种我常用的组合。

## 8.1 关闭程序

在每次打完 LOL 的时候，基地爆炸我们常常会发现有很多玩家直接退出了游戏，我们比较好奇，为什么他们能这么地退出了游戏，按理来说退出游戏需要等待 5s 确认。答案揭晓了，就是使用了 <kbd>Alt + F4</kbd> 的组合键实现了快速退出游戏。具体演示如 [Fig.31](#Fig.31) 所示。

<a id=Fig.31></a>
<div align=center>
    <img src=./imgs_markdown/gif.31.gif
    width=80%>
    <center>Fig.31 Alt + F4 退出程序的示例</center>
</div></br>

> 💡  很多程序和界面都是可以通过 <kbd>Alt + F4</kbd> 实现退出的，当然有些是不支持的。

## 8.2 程序间的切换

我们在使用电脑的时候一般会使用多个软件，那么这些软件应该如何进行切换呢？我们可以使用鼠标左键在任务栏中点击不同的任务，但如果这个程序是全屏状态呢？此时任务栏是隐藏的，又如何切换程序呢？

我们可以使用 <kbd>Alt + Tab</kbd> 来切换不同的任务，具体演示如 [Fig.32](#Fig.32) 所示。

<a id=Fig.32></a>
<div align=center>
    <img src=./imgs_markdown/gif.32.gif
    width=80%>
    <center>Fig.32 Alt + Tab 实现程序间切换的示例</center>
</div></br>

## 8.3 竖向选择文字

我们在编辑文字的时候，无论使用鼠标左键还是 <kbd>Shift</kbd> 进行文字选择的时候，都是横向选择，那么我们可以进行竖向选择吗？当然是可以的，使用 <kbd>Shift + Alt + 鼠标左键</kbd> 按键即可实现竖向选择，具体演示如 [Fig.33](#Fig.33) 所示。

<a id=Fig.33></a>
<div align=center>
    <img src=./imgs_markdown/gif.33.gif
    width=80%>
    <center>Fig.33 Alt + Shift + 鼠标左键实现竖向选择多行的示例</center>
</div></br>

## 8.4 任意选择多行文字

我们也可以使用 <kbd>Alt + 鼠标左键</kbd> 选择任意行，具体演示如 [Fig.34](#Fig.34) 所示。

<a id=Fig.34></a>
<div align=center>
    <img src=./imgs_markdown/gif.34.gif
    width=80%>
    <center>Fig.34 Alt + 鼠标左键实现任意选择多行文字的示例</center>
</div></br>

# 9. Tab

## 9.1 缩进

<kbd>Tab</kbd> 是 Table 的缩写，而 Table 的意思是表格，那么在不同的程序中，<kbd>Tab</kbd> 按键也有有着不同的作用。比如：

- 在 Python 中（具体演示如 [Fig.35](#Fig.35) 所示）：
  - 在一行文字的开头，一个 <kbd>Tab</kbd> 等于 4 个空格。
  - 选中多行代码，一个 <kbd>Tab</kbd> 等于向右进行一次缩进。
- 在 Word 中（具体演示如 [Fig.36](#Fig.36) 所示）：
  - 在一行文字的开头，一个 <kbd>Tab</kbd> 等于 1 个缩进。
  - 选中多行文字，一个 <kbd>Tab</kbd> 等于向右进行一次缩进。
- 在 Markdown 中（具体演示如 [Fig.37](#Fig.37) 所示）：
  - 在一行文字的开头，一个 <kbd>Tab</kbd> 等于 1 个缩进。
  - 选中多行文字，一个 <kbd>Tab</kbd> 等于向右进行一次缩进。

<a id=Fig.35></a>
<div align=center>
    <img src=./imgs_markdown/gif.35.gif
    width=80%>
    <center>Fig.35 在 Python 中 Tab 实现缩进的示例</center>
</div></br>

<a id=Fig.36></a>
<div align=center>
    <img src=./imgs_markdown/gif.36.gif
    width=80%>
    <center>Fig.36 在 Word 中 Tab 实现缩进的示例</center>
</div></br>

<a id=Fig.37></a>
<div align=center>
    <img src=./imgs_markdown/gif.37.gif
    width=80%>
    <center>Fig.37 在 Markdown 中 Tab 实现缩进的示例</center>
</div></br>

## 9.2 反向缩进

既然 <kbd>Tab</kbd> 是缩进，那么 <kbd>Shift + Tab</kbd> 就是反向缩进，和缩进的含义相反，这里就不再演示了。

## 9.3 切换下一个项目

在网页中输入密码，比如：

```
账号：
密码：
```

一般情况下，我们输入完账号后，需要按鼠标左键点击“密码”这一栏才可以输入密码，其实还有一种更加优雅和快速的方法，按下 <kbd>Tab</kbd> 即可快速切换到“密码”这一栏，具体演示如 [Fig.38](#Fig.38) 所示。

<a id=Fig.38></a>
<div align=center>
    <img src=./imgs_markdown/gif.38.gif
    width=80%>
    <center>Fig.38 Tab 实现切换下一个项目的示例</center>
</div></br>

> <kbd>Tab</kbd> 在切换 item 的应用有很多，不同的程序、网页都有不同的逻辑，大家可以自行探索。

# 10. Windows

<kbd>Windows</kbd> 键大家可能比较陌生，但其实 <kbd>Windows</kbd> 非常强大，不管是在 Windows 操作系统还是 Linux 操作系统中，<kbd>Windows</kbd> 都有着非常好用的功能。

## 10.1 切换桌面

我们是否有这样的场景，当我们在工作的时候突然来了其他任务，此时任务栏已经差不多快满了，如果两个不同的任务放在一起，可能比较混乱。微软当然也想到了这样的场景，因此微软提供了多桌面的功能。

我们可以通过切换不同的桌面，从而在不同的桌面专门处理特定的任务，这样可以大大提高我们的工作效率。我们可以使用 <kbd>Windows + Tab</kbd> 来切换桌面，具体演示如 [Fig.39](#Fig.39) 所示。

<a id=Fig.39></a>
<div align=center>
    <img src=./imgs_markdown/2024-04-04-21-24-21.png
    width=80%>
    <center>Fig.39 Windows + Tab 实现切换不同桌面的示例</center>
</div></br>

从 [Fig.39](#Fig.39) 中我们可以看到，此时我们有三个桌面，并且我们也可以新增桌面。我们也可以将不同的程序拖动到不同的桌面下，从而实现在不同桌面处理不同任务。

> 因为无法录制到，所以这里使用截图进行展示。

## 10.2 快速打开任务栏中的任务

[Fig.40](#Fig.40) 是一个任务栏，我们想要选择指定的任务，一般使用鼠标左键。还有一种方式就是使用 <kbd>Windows + 数字</kbd> 的方式进行快速选择任务栏指定位置的程序。比如：

- <kbd>Windows + 1</kbd>：快速打开任务栏第一个任务（我这里是 Edge 浏览器）
- <kbd>Windows + 2</kbd>：快速打开任务栏第二个任务（我这里是 VSCode 浏览器）
- ...

<a id=Fig.40></a>
<div align=center>
    <img src=./imgs_markdown/2024-04-04-21-26-46.png
    width=80%>
    <center>Fig.40 任务栏的示例</center>
</div></br>

那么使用 <kbd>Windows + 数字</kbd> 有什么好处呢？拿我举例子，我把 Edge 浏览器固定在了任务栏的第一个位置，所以我每次想打开浏览器，直接按 <kbd>Windows + 1</kbd> 即可，非常高效和优雅。

## 10.3 搜索

细心的同学会发现我的任务栏是没有搜索按钮的，可能有些同学会觉得很奇怪，明明搜索按钮很好用，我为什么不用呢？其实我是使用的，只不过我将其进行了隐藏，并使用 <kbd>Windows + S</kbd> 的快捷键呼出搜索，具体演示如 [Fig.41](#Fig.41) 所示。

<a id=Fig.41></a>
<div align=center>
    <img src=./imgs_markdown/gif.41.gif
    width=80%>
    <center>Fig.41 Windows + S 唤醒“搜索”的示例</center>
</div></br>

这样我们就没必要把搜索框留在我们的任务栏里了，大大加强了任务栏的空间利用率。

> 💡  `S = Search`

## 10.4 回到桌面

这个功能是很多打工人必备的，毕竟可以一键摸鱼 🤣。这个按键就是 <kbd>Windows + D</kbd>，它可以瞬间让我们回到桌面。这个快捷键等价于任务栏右下角的“显示桌面”。

> ⚠️  注意：在全屏程序，如 LOL 中，<kbd>Windows + D</kbd> 是不起作用的！

> 💡  `D = Desktop`

## 10.5 快速打开文件资源管理器

和 <kbd>Windows + E</kbd> 的组合键可以快速打开资源管理器，具体演示如 [Fig.42](#Fig.42) 所示。

<a id=Fig.42></a>
<div align=center>
    <img src=./imgs_markdown/gif.42.gif
    width=80%>
    <center>Fig.42 Windows + E 快速打开资源管理器的示例</center>
</div></br>

> 💡  `E = Explorer`

## 10.6 快速锁屏

吃午饭或者下班了之后我们为了防止有老六动我们的电脑，我们使用 <kbd>Windows + L</kbd> 锁定电脑，解锁需要输入我们的用户密码（和开机是一样的）。

> 💡  `L = Lock`

## 10.7 快速对齐窗口

比如我们想要让屏幕一分为二，左边浏览网页，右边记录，那么我们可以使用 <kbd>Windows + ←/→</kbd> 帮助我们快速对齐窗口，具体演示如 [Fig.43](#Fig.43) 所示。

<a id=Fig.43></a>
<div align=center>
    <img src=./imgs_markdown/gif.43.gif
    width=80%>
    <center>Fig.43 Windows + ←/→ 快速对齐窗口的示例</center>
</div></br>