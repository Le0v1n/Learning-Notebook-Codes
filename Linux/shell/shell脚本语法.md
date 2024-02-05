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

```sh
#!/bin/sh
echo "Hello World !"
```




