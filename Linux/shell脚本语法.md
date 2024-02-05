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

|执行方式|要求|
|:-|:-|
|绝对路径/相对路径|必须是可执行文件|
|sh|无（普通文件和脚本文件都可以）|
|source|无（普通文件和脚本文件都可以）|

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
    <img src=./imgs_markdown/2024-02-05-17-49-36.png
    width=100%>
    <center></center>
</div>

那么为什么前两种方式没有打印出来 `uname` 变量呢？