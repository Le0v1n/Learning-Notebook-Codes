# 1. Shell 编程概述

## 1.1 Shell 名词解释

在 Linux 操作系统中，Shell 是一个命令行解释器，它为用户提供了一个与操作系统内核交互的界面。用户可以通过 Shell 输入命令，然后 Shell 将这些命令翻译给操作系统去执行。Shell 还可以用来编写脚本，自动化执行重复的任务。

- kernel：Linux 的内容，主要是为了和硬件打交道。
- Shell：
  - 命令解释器（command interpreter）
  - Shell 是一个用 C 语言编写的程序，它是用户使用 Linux 的桥梁。Shell 既是一种命令语言，又是一种程序设计语言。
  - Shell 是指一种应用程序，这个应用程序提供了一个界面，用户通过这个界面访问操作系统内核的服务。

Shell 目前有两大主流：
1. sh：
   1. Bourne Shell (sh): Solaris, hpux 默认的 Shell
   2. Bourne again shll (bash)：Linux 系统默认的 Shell
2. csh：
   1. C shell (csh)
   2. tc shell (tcsh)

⚠️ 我们在使用 shell 脚本时需要进行声明，使用 `#!`，从而告诉系统我们要使用哪个路径下的 shell 解释器，例子如下：

```bash
#!/bin/bash
echo "Hello World!"
```

<kbd><b>Q</b>uestion</kbd>：`#!/bin/sh` 和 `#!/bin/bash` 有区别吗？

<kbd><b>A</b>nswer</kbd>：`#!/bin/sh` 和 `#!/bin/bash` 的区别主要在于它们指定了不同的 shell 解释器。

- `#! /bin/sh` 是一个较老的用法，它指定脚本应该由 Bourne shell（或者兼容 Bourne shell 的 shell）来执行。在大多数现代 Linux 系统中，`/bin/sh` 已经是 `bash`（Bourne-Again SHell）的一个符号链接或者是一个兼容 shell 的链接，因为 `bash` 兼容并扩展了原始的 Bourne shell。所以，即使你使用 `#!/bin/sh`，实际执行的可能是 `bash`。
`#!/bin/bash` 则明确指定了脚本应该由 `bash` 来执行。这样做的好处是可以确保使用 `bash` 的特定扩展和特性。如果你在脚本中使用了 `bash` 的特定功能，那么你应该使用 `#! /bin/bash` 作为 shebang。

## 1.2 Shell 脚本的执行方式

### 1.2.1 输入脚本的绝对路径或相对路径

```bash
# 绝对路径
/mnt/f/Projects/本地代码/Learning-Notebook-Codes/Linux/shell/codes/hello_world.sh

# 相对路径
Linux/shell/codes/hello_world.sh
```

⚠️ 该种执行方式下，`.sh` 文件必须是一个可执行文件，即拥有 `x` 权限

### 1.2.2 `bash` 或者 `sh` + 脚本路径

```bash
# 绝对路径
sh /mnt/f/Projects/本地代码/Learning-Notebook-Codes/Linux/shell/codes/hello_world.sh

# 相对路径
sh Linux/shell/codes/hello_world.sh
```

⚠️ 当脚本没有 `x` 权限时，root 用户和文件所有者可以通过该方式正常执行

### 1.2.3 `source` + 脚本路径

```bash
# 绝对路径
source /mnt/f/Projects/本地代码/Learning-Notebook-Codes/Linux/shell/codes/hello_world.sh

# 相对路径
source Linux/shell/codes/hello_world.sh
```

### 1.2.4 三种执行方式的对比

| 执行方式          | 要求                           |
| :---------------- | :----------------------------- |
| 绝对路径/相对路径 | 必须是可执行文件               |
| sh                | 无（普通文件和脚本文件都可以） |
| source            | 无（普通文件和脚本文件都可以） |

不同点：我们举个例子，现在终端中输入下面的命令

```bash
uname = admin
echo $uname
```

之后修改 `Linux/shell/codes/hello_world.sh` 脚本：

```bash
#!/bin/bash

echo "Hello World!"
echo $uname
```

我们再看一下三种方式的结果：

<div align=center>
    <img src=./imgs_markdown/2024-02-05-20-52-34.png
    width=100%>
    <center></center>
</div>

那么为什么前两种方式没有打印出来 `uname` 变量呢？

首先，我们需要了解这三种执行方式的区别：
1. **绝对路径/相对路径**：这种方式会开启一个子 Shell 来执行脚本。这意味着脚本将在一个全新的环境中运行，其中的变量和操作不会影响到父 Shell。
2. **sh**：这种方式也是在一个新的 Shell 环境中执行脚本，与绝对路径/相对路径执行方式类似。
3. **source**：这种方式会在当前 Shell 环境中执行脚本，而不是在一个新的子 Shell 中。因此，脚本中的任何更改（例如，设置或更改环境变量）都会影响到当前 Shell。

## 1.3 export 关键字

在 Shell 脚本中，`export` 关键字用来将脚本中的变量声明为环境变量，这意味着该变量可以被脚本中执行的任何命令或子 Shell 访问。环境变量对于定义和控制脚本的执行环境非常有用，例如，可以用来设置路径、定义配置选项或传递参数。

当你在 Shell 脚本中使用 `export` 时，你实际上是修改了当前 Shell 的环境。任何从这个 Shell 启动的子 Shell 或者进程都会继承这些环境变量。

下面是一个简单的例子：
```bash
#!/bin/bash
# 定义一个变量
my_variable="Hello, World! "

# 输出变量，此时它仅存在于当前 Shell 中
echo $my_variable

# 将变量导出为环境变量
export my_variable

# 此时，启动一个子 Shell
bash -c 'echo $my_variable'
```

结果如下：

<div align=center>
    <img src=./imgs_markdown/2024-02-05-21-48-59.png
    width=100%>
    <center></center>
</div>

在上面的脚本中，`my_variable` 首先被定义并输出，然后通过 `export` 导出。在启动的子 Shell 中，我们同样可以访问 `my_variable`，这是因为 `export` 使得该变量成为了环境变量。

需要注意的是，默认情况下，在函数内部设置的变量是不会自动成为环境变量的，如果需要在函数外部访问函数内部设置的变量，也需要使用 `export` 来导出这些变量。

⚠️ 这个环境变量只是在当前终端生效，如果我们在 A 终端使用了 `export`，但在新开启的终端 B 中是不生效的，如下所示：

<div align=center>
    <img src=./imgs_markdown/2024-02-05-21-51-56.png
    width=100%>
    <center></center>
</div>

<div align=center>
    <img src=./imgs_markdown/2024-02-05-21-52-06.png
    width=100%>
    <center></center>
</div>

这是因为每个终端会话（session）通常都有自己的独立环境。当我们打开一个新的终端时，它会从系统环境中继承一系列默认的环境变量，但不会继承其他终端会话中通过 `export` 设置的变量。

💡 如果我们希望环境变量在新的终端会话中自动生效，我们可以将 `export` 命令添加到我们的 Shell 启动文件中，例如 `~/.bashrc`、`~/.bash_profile` 或 `~/.profile`，这取决于我们使用的 Shell 和配置。这样，每次启动新的 Shell 时，这些文件都会被读取，从而设置相应的环境变量。

## 1.4 cat 关键字

在 Shell 中，`cat`（全称“concatenate”）是一个常用的命令行工具，用于查看文件内容、创建文件、文件合并以及将文件内容重定向到另一个文件等操作。

### 1.4.1 查看文件内容

假设 `A.txt` 文件中的内容如下：

```
这是A文件
```

那我们可以直接在 Shell 中查看它：

```bash
cat cat Linux/shell/assets/A.txt
```

结果如下：

<div align=center>
    <img src=./imgs_markdown/2024-02-05-21-58-25.png
    width=100%>
    <center></center>
</div>

❓**Question**：我们会发现并没有换行，这是为什么呢？

🅰️ **Answer**：这是因为我们的 `A.txt` 文件的内容确实以 `这是A文件` 结尾，并且后面没有换行符，那么 `cat` 命令的输出中也就不会显示换行。在文本文件中，换行符通常是一个特殊的字符，它告诉文本编辑器或命令行界面在新的一行开始。如果文件最后一行没有换行符，那么命令提示符就会直接跟在文件内容的后面，看起来就像是没有换行一样。

那我们在 `A.txt` 文件后面添加一行呢？即 `A.txt` 文件内容如下：

```
这是A文件

```

结果如下：

<div align=center>
    <img src=./imgs_markdown/2024-02-05-22-00-22.png
    width=100%>
    <center></center>
</div>

正常换行了，那我们还有其他方法吗？比如我们在 `A.txt` 文件的末尾加上一个换行符：

```
这是A文件\n
```

我们再看一下效果：

<div align=center>
    <img src=./imgs_markdown/2024-02-05-22-01-23.png
    width=100%>
    <center></center>
</div>

⚠️ 并没有换行！这是因为我们在 `A.txt` 文件的末尾添加一个文本字符串 `"\n"`，**这并不会在文件中添加一个真正的换行符**。`"\n"` 只是一个由两个字符组成的普通字符串：反斜杠 (`\`) 和字母 `n`。

> 💡 如果要创建一个真正的换行符，我们需要在文本编辑器中实际按下 Enter 键，或者在命令行中使用转义序列。

在命令行中，我们可以使用 `echo` 命令的 `-e` 选项来解释转义序列。下面是如何在 `A.txt` 文件末尾添加一个真正的换行符：

```bash
echo -e "这是A文件\n" >> Linux/shell/assets/A.txt
```

这个命令会在 `A.txt` 文件中追加文本 `这是A文件` 后跟一个真正的换行符。

如果我们想在不使用 `-e` 选项的情况下添加换行符，可以在 `A.txt` 文件中直接按 Enter 键，或者在命令行中使用 `printf` 命令：

```bash
printf "这是A文件\n" >> Linux/shell/assets/A.txt
```

`printf` 命令默认就会解释转义序列，所以不需要额外的选项来添加换行符。

> ⚠️ 注意，如果我们直接在文本编辑器中编辑文件并保存，通常编辑器会在文件的末尾自动添加一个换行符，除非我们明确地删除了它。

### 1.4.2 创建新文件或编辑现有文件

```bash
# cat > filename
cat > Linux/shell/assets/A.txt
```

<div align=center>
    <img src=./imgs_markdown/2024-02-05-22-04-44.png
    width=100%>
    <center></center>
</div>

此时我们可以输入我们想输入的内容，在按下 `Ctrl+D` 保存并退出。

<div align=center>
    <img src=./imgs_markdown/2024-02-05-22-06-26.png
    width=100%>
    <center></center>
</div>

<div align=center>
    <img src=./imgs_markdown/2024-02-05-22-07-02.png
    width=100%>
    <center></center>
</div>

⚠️ ==如果文件已存在，这将清空文件内容==


❓**Question**：`cat > filename` 只能清空所有内容并编写吗？可以追加内容吗？

🅰️ **Answer**：当然是可以的了，往下看。

### 1.4.3 将文件内容追加到另一个文件的末尾

```bash
cat file1 >> file2
```

这会将`file1`的内容追加到`file2`的末尾，而不是覆盖它。

举个例子：

```bash
cat >> Linux/shell/assets/A.txt
```

<div align=center>
    <img src=./imgs_markdown/2024-02-05-22-12-08.png
    width=100%>
    <center></center>
</div>

💡 **总结一下**：

- `cat > filename`：清空文件内容并重新编写。
- `cat >> filename`：追加内容到文件的末尾。


### 1.4.4 将多个文件的内容合并输出

```bash
cat file1 file2 file3
```

这将依次显示 `file1`、`file2` 和 `file3` 的内容。

我们实际试一下：

```bash
cat Linux/shell/assets/A.txt Linux/shell/assets/B.txt Linux/shell/assets/C.txt
```

<div align=center>
    <img src=./imgs_markdown/2024-02-05-22-13-31.png
    width=100%>
    <center></center>
</div>

### 1.4.5 将文件内容重定向（redirection）到另一个文件

```bash
cat file1 > file2
```

这会将 `file1` 的内容覆盖到 `file2` 中。如果 `file2` 不存在，则创建它。

例子：

```bash
cat Linux/shell/assets/A.txt > Linux/shell/assets/exp-redirection.txt
```

<div align=center>
    <img src=./imgs_markdown/2024-02-05-22-16-37.png
    width=100%>
    <center></center>
</div>

> 💡 如果我们想让 file1 的内容追加到 file2 的末尾，使用 `cat file1 >> file2` 也是可以的！

### 1.4.6 显示文件内容并显示行号

```bash
cat -n filename
```

试一下：

```bash
cat -n Linux/shell/assets/A.txt
```

<div align=center>
    <img src=./imgs_markdown/2024-02-05-22-17-25.png
    width=100%>
    <center></center>
</div>

### 1.4.7 总结

```bash
# 1. 查看文件内容
cat filename

# 2. 创建新文件或编辑现有文件（使用 ctrl+D 保存）
cat > filename

# 3. 将文件内容追加到另一个文件的末尾
cat >> filename

# 4. 将多个文件的内容合并输出
cat file1 file2 file3

# 5. 将文件内容重定向（redirection）到另一个文件
cat file1 > file2

# 6. 显示文件内容并显示行号
cat -n filename
```

> ⚠️ `cat` 命令在处理小文件或查看文件内容时非常有用，但它不适合处理非常大的文件，因为`cat`会一次性将整个文件加载到内存中，这可能会导致内存不足的问题。
> 
> 💡 对于大文件，可以使用其他命令如`less`或`more`，这些命令可以分页显示文件内容，避免内存问题。

# 2. Shell 基础入门

## 2.1 shell 变量

### 2.1.1 变量命名

定义变量时，变量名不加美元符号 `$`，且有一些命名规则：

- 命名只能使用英文字母、数字和下划线（`_`），且首个字符必须是非数字的字母
- 中间不能有空格，但可以使用下划线 `_` 作为单词分隔符
- 不能使用标点符号，如 `!`, `@`, `#`, `$`, `%`, `^`, `&`, `*`, `(`, `)`, `-`, `=`, `+`, `[`, `]`, `{`, `}`, `|`, `:`, `;`, `'`, `"`, `<`, `>`, `,`, `.`, `/`, `?`
- 不能使用 `bash` 里的关键字，如 `if`, `for`, `while`, `function` 等

> ⚠️ 不推荐使用 `-`，因为它在某些情况下可能会导致语法错误或被解释为命令行选项。在变量替换时，如果变量名包含 `-`，建议将变量名用大括号 `{}` 包起来，例如 `${var-value}`。

### 2.1.2 变量类型

- **局部变量**：在脚本或命令中定义，仅在当前 shell 实例中有效，其他 shell 启动的程序不能访问局部变量。
- **环境变量**：任何由当前 shell 启动的程序都可以访问环境变量。环境变量对于程序的配置和运行非常重要。
- **shell 变量**：由 shell 程序设置的特殊变量，包括环境变量和局部变量。Shell 变量通常用于存储 shell 的状态信息或作为脚本编程的辅助工具。

### 2.1.3 补充说明

- 对于环境变量，可以使用 `export` 命令将局部变量提升为环境变量，使得它对当前 shell 及其所有子进程可见。
- ⚠️ 变量赋值时，等号 `=` 两边不能直接接空格。
- 为了避免潜在的问题，建议在脚本中使用明确的变量名，避免使用可能具有特殊含义的字符。

# 参考

1. 【这可能是B站讲的最好的Linux Shell脚本教程，3h打通Linux-shell全套教程，从入门到精通完整版】https://www.bilibili.com/video/BV1Eq4y1z7u8?p=4&vd_source=ac73c03faf1b37a5bc2296969f45cf7b




