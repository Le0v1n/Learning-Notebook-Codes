学习一门语言的顺序如下：

<div align=center>
    <img src=./imgs_markdown/plots-学习一门语言的顺序.jpg
    width=100%>
    <center></center>
</div>

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

<kbd><b>Q</b>uestion</kbd>：我们会发现并没有换行，这是为什么呢？

<kbd><b>A</b>nswer</kbd>：这是因为我们的 `A.txt` 文件的内容确实以 `这是A文件` 结尾，并且后面没有换行符，那么 `cat` 命令的输出中也就不会显示换行。在文本文件中，换行符通常是一个特殊的字符，它告诉文本编辑器或命令行界面在新的一行开始。如果文件最后一行没有换行符，那么命令提示符就会直接跟在文件内容的后面，看起来就像是没有换行一样。

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

由于我们没有敲回车，所以 `(base) root@Le0v1n...` 这些内容在我们输入的内容之后。如果我们想要换行的效果，可以在输入内容的之后手动敲回车，如下所示：

<div align=center>
    <img src=./imgs_markdown/2024-02-05-22-07-02.png
    width=100%>
    <center></center>
</div>

⚠️ ==如果文件已存在，这将清空文件内容==


<kbd><b>Q</b>uestion</kbd>：`cat > filename` 只能清空所有内容并编写吗？可以追加内容吗？

<kbd><b>A</b>nswer</kbd>：当然是可以的了，往下看。

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

### 2.1.4 示例

```bash
#!/bin/bash

# 定义变量
name="Le0v1n"

# 变量的调用
echo $name
echo ${name}
```

```
Le0v1n
Le0v1n
```

> 💡 使用花括号 `{}` 来包围变量名在某些情况下是必要的，比如变量名后面紧跟着其他字符时，或者变量名是一个复杂表达式的一部分时，它可以避免语法歧义，确保Bash正确地解析变量名。

## 2.2 Shell 的字符串

### 2.2.1 定义

字符串是 Shell 编程中最常用也是最有用的数据类型之一。在 Shell 中，字符串可以使用单引号，也可以使用双引号，而且确实在某些情况下可以不使用引号。

### 2.2.2 单双引号

下面是关于单引号和双引号的详细说明：

- **单引号**：
  - 单引号里的任何字符都会原样输出，不会进行变量替换或特殊字符的转义。
  - 单引号字符串中的变量是无效的，即它们不会被替换为变量的值。
  - 单引号字符串中不能直接出现单独一个的单引号，但可以通过成对出现的方式包含单引号，例如 `'I\'m a string'`。
  
- **双引号**：
  - 双引号里可以有变量，Shell 会将变量的值替换到字符串中。
  - 双引号里可以出现转义字符，例如 `"\n"` 会被转义为换行符。

关于不使用引号的情况，Shell 解释器会根据一些规则来解释未加引号的字符串。通常，未加引号的字符串会被视为命令的参数，并且可能会发生单词分割和文件名扩展。**这意味着如果字符串中包含空格或特殊字符，Shell 可能会错误地解释它们**。因此，尽管在某些简单的情况下可以省略引号，但**为了确保字符串的正确解析，通常建议使用单引号或双引号来定义字符串**。

### 2.2.3 单双引号的示例

```bash
#!/bin/bash

str1="hello world 1"  # 双引号
str2='hello world 2'  # 单引号

# 直接调用
echo $str1  # hello world 1
echo $str2  # hello world 2
```

```bash
#!/bin/bash

# 字符串拼接：双引号
name='le0v1n'
name1="hello, $name!"    # 双引号可以转义
name2="hello, "$name"!"  # "hello, "是成对的，"!"是成对的，所以$name不受影响
name3="hello, '$name'!"  # "hello, '$name'!"是成对的，且双引号可以转义
name4="hello, ${name}!"  # "hello, ${name}!"是成对的，且双引号可以转义

echo $name1  # hello, le0v1n!
echo $name2  # hello, le0v1n!
echo $name3  # hello, 'le0v1n'!
echo $name4  # hello, le0v1n!
```

```bash
#!/bin/bash

# 字符串拼接：单引号
name='le0v1n'

name1='hello, $name!'    # 'hello, $name!'是成对的，但单引号不能转义
name2='hello, '$name'!'  # 'hello, '是成对的，'!'是成对的，所有$name不受影响
name3='hello, "$name"!'  # 'hello, "$name"!'是成对的，但单引号不能转义
name4='hello, ${name}!'  # 'hello, ${name}!'是成对的，但单引号不能转义

echo $name1  # hello, $name!
echo $name2  # hello, le0v1n!
echo $name3  # hello, "$name"!
echo $name4  # hello, ${name}!
```

### 2.2.4 获取字符串长度和字符串切片

```bash
#!/bin/bash

# 字符串长度
email="Le0v1n@163.com"

echo ${email}      # Le0v1n@163.com
echo ${#email}     # 14
echo ${email:0:5}  # Le0v1
```

> ⚠️ 在运行该脚本的时候，要使用 `bash` 或者 `./` 或者 `source`，不要使用 `sh`

## 2.3 Shell 数组

### 2.3.1 定义

Shell 数组在 Bash 中是一个强大的特性，它允许我们存储多个值在单个变量中。数组在 Bash 中以索引方式访问，这使得它们在处理序列时非常有用。

- bash 支持一维数组（不支持多维数组），并且没有限定数组的大小
- 数组元素的索引从 0 开始。获取数组中的元素要使用索引，索引可以是整数或算术表达式，其值应该 ≥ 0

### 2.3.2 数组的创建

Shell 数组的定义：括号用来表示数组，数组元素用空格符号分隔开。

```bash
数组名=(value1 value2 ... valueN)
```

### 2.3.3 数组的索引

```bash
#!/bin/bash

# 创建一个shell数组
exp_array=( "Hello" "world" "你好" '单引号')

echo "数组为: ${exp_array}"     # 默认输出第一个元素
echo "数组为: ${exp_array[0]}"  # 按索引来
echo "数组为: ${exp_array[1]}" 
echo "数组为: ${exp_array[2]}"
echo "数组为: ${exp_array[3]}"
echo "数组为: ${exp_array[4]}"  # 越界的直接打印空
```

```
数组为: Hello
数组为: Hello
数组为: world
数组为: 你好
数组为: 单引号
数组为: 
```

### 2.3.4 取出所有元素

```bash
#!/bin/bash

exp_array=( "Hello" "world" "你好" '单引号')

# 使用 @ 符号可以取出数组中所有元素
echo ${exp_array[@]}
```

```
Hello world 你好 单引号
```

### 2.3.5 获取数组的长度

```bash
#!/bin/bash

# 获取数组的长度
array_len_1=${#exp_array[@]}
array_len_2=${#exp_array[*]}

echo $array_len_1
echo ${array_len_2}
```

```
4
4
```

> ⚠️ 注意：`${#exp_array[@]}` 和 `${#exp_array[*]}` 都可以用来获取数组的长度，它们都会输出数组的长度。`${#exp_array[@]}` 适用于只提供数组名称的情况，而 `${#exp_array[*]}` 适用于提供数组元素列表的情况下。

### 2.3.6 获取数组中某一个元素的长度

```bash
#!/bin/bash

exp_array=( "Hello" "world" "你好" '单引号')

# 获取数组中某一个元素的长度
elem_len_0=${#exp_array[0]}
elem_len_1=${#exp_array[1]}
elem_len_2=${#exp_array[2]}
elem_len_3=${#exp_array[3]}
elem_len_4=${#exp_array[4]}
echo ${elem_len_0}  # 5
echo ${elem_len_1}  # 5
echo ${elem_len_2}  # 2
echo ${elem_len_3}  # 3
echo ${elem_len_4}  # 0
```

> ⚠️ 注意：数组索引是从 0 开始的，所以 `${exp_array[4]}` 是无效的，因为它超出了数组的范围。在尝试访问数组元素时，确保索引在数组范围内。

## 2.4 Shell 的注释

在 Shell 脚本中，注释用于解释代码，帮助其他开发者理解脚本的功能，或者在脚本中暂时禁用某些代码行。Shell 脚本中的注释有以下几种方式：

### 2.4.1 单行注释

使用 `#` 符号开始一个单行注释。`#` 符号后面的内容直到行尾都会被解释器忽略。

```bash
# 这是一个单行注释
```

### 2.4.2 多行注释（块注释）
虽然 Shell 没有专门的多行注释语法，但是你可以使用 `#` 符号来开始一个多行注释块。这种注释块会一直持续到遇到另一个 `#` 符号。

```bash
# 这是一个多行注释的开始
# 你可以在这里写很多行注释
# 直到再次遇到 #
```

```bash
# ----------------------------------------------
# 这也是一个多行注释
# 这也是一个多行注释
# 这也是一个多行注释
# 这也是一个多行注释
# 这也是一个多行注释
# ----------------------------------------------

##### 服务器配置-start #####
# comment
# comment
# comment
# comment
# comment
##### 服务器配置-end   #####
```

### 2.4.3 Here Document（文档字符串）
Here Document 是一种特殊类型的注释，它可以用于将输入传递给命令。虽然它主要用于输入，但也可以用于创建注释块。

```bash
cat <<'EOF'
# 这是一个多行注释的开始
# 你可以在这里写很多行注释
# 直到再次遇到 #
EOF
```

在编写 Shell 脚本时，建议合理使用注释，以提高代码的可读性和可维护性。注释应该清晰、简洁，并且与代码保持一致。

## 2.5 Shell 脚本传参

在 Shell 脚本中，参数传递是脚本与调用脚本的环境之间交互的一种方式。参数可以是一系列值，传递给脚本以便脚本在执行时可以使用这些值。

### 2.5.1 参数传递机制

Shell 脚本可以通过命令行从外部接收参数。这些参数可以在脚本内部通过 `$1`、`$2`、`$3` 等特殊变量来访问。这些特殊变量被称为位置参数，它们按顺序存储了传递给脚本的参数值。

### 2.5.2 位置参数

| 特殊变量 | 说明 |
| :----: | :--: |
| `$0` | 脚本的名称 |
| `$1` | 第一个参数 |
| `$2` | 第二个参数 |
| `$3` | 第三个参数 |
| ... | ... |
| `$n` | 第 n 个参数 |

### 2.5.3 获取所有参数

可以使用 `$@` 或 `$*` 特殊变量来获取所有传递给脚本的参数。

- `$@`：表示所有位置参数的列表，每个参数都作为单独的单词处理。
- `$*`：与 `$@` 类似，但它将所有参数视为一个单词。

### 2.5.4 获取参数个数

可以使用 `$#` 特殊变量来获取传递给脚本的所有参数的个数。

### 2.5.5 示例

下面是一个简单的 Shell 脚本示例，展示了如何接收和处理参数。

```bash
#!/bin/bash

# 打印脚本名称
echo "脚本名称: $0"

# 打印所有参数
echo "所有参数: $@"

# 打印参数个数
echo "参数个数: $#"

# 打印第一个参数
echo "第一个参数: $1"

# 打印第二个参数
echo "第二个参数: $2"

# 打印第三个参数
echo "第三个参数: $3"
```

要运行这个脚本并传递参数，可以使用以下命令：

```bash
./script_name.sh 参数1 参数2 参数3
```

<div align=center>
    <img src=./imgs_markdown/2024-02-06-16-19-12.png
    width=100%>
    <center></center>
</div>

### 2.5.6 特殊参数

除了位置参数之外，还有一些特殊的参数，它们以 `$` 符号开始，例如：

- `$*` 和 `$@`：用于引用所有位置参数。
- `$#`：传递给脚本的所有参数的个数。
- `$?`：最后一次命令的退出状态。
- `$$`：当前 Shell 进程 ID（PID）。

这些特殊参数在脚本编程中非常有用，尤其是在需要与脚本调用者交互或处理命令行输入时。

### 2.5.7 示例：脚本使用特殊参数

下面是一个使用特殊参数的示例脚本：

```bash
#!/bin/bash

# 打印所有参数，使用 $* 和 $@
echo "所有参数: $*"
echo "所有参数: $@"

# 打印参数个数
echo "参数个数: $#"

# 打印上一个命令的退出状态
echo "上一个命令的退出状态: $?"

# 打印当前 Shell 的进程 ID
echo "当前 Shell 的进程 ID: $$"
```

<div align=center>
    <img src=./imgs_markdown/2024-02-06-16-20-57.png
    width=100%>
    <center></center>
</div>

这个脚本演示了如何使用特殊参数来获取关于脚本参数和执行环境的各种信息。

# 参考

1. [这可能是B站讲的最好的Linux Shell脚本教程，3h打通Linux-shell全套教程，从入门到精通完整版](https://www.bilibili.com/video/BV1Eq4y1z7u8)