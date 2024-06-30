
# 1. WSL2 安装

⚠️  <font color='red'><b>先决条件</b></font>：Windows 10 版本需要 2004 及更高版本（内部版本 19041 及更高版本）或 Windows 11 才能使用以下命令。

```bash
wsl --install
```

WSL会自动安装，安装完毕后它会提示我们要重启电脑。

# 2. 优化

重启完电脑后，我们需要对WSL做出一系列的优化。

## 2.1 更改 WSL 所在路径

安装完 WSL 后，默认是在 C 盘下的，一般来说系统盘的容量通常来说是有限的，需要更改安装目录。

1. 查看 WSL 的运行状态:
   ```bash
   wsl -l -v
   ```
2. 确保 WSL 处于关闭状态（Stopped），如果是 Running，则需要关闭：
   ```bash
   wsl --shutdown
   ```
   之后再次查询 WSL 状态
3. 导出当前的 Linux 系统镜像:
   ```bash
    wsl --export Ubuntu d:\image_ubuntu.tar
   ```
   之后会在 D 盘中有一个大小为 1.2G 的 `image_ubuntu.tar` 文件
4. 移除之前注册的 WSL：
   ```bash
   wsl --unregister Ubuntu
   ```
5. 再次输入查看 WSL 的运行状态：
   ```bash
   wsl -l -v
   ```
   ```
   适用于 Linux 的 Windows 子系统没有已安装的分发版。
   可以通过访问 Microsoft Store 来安装分发版:
   https://aka.ms/wslstore
   ```
   这样说明我们取消注册成功了
6. 我们重新注册 WSL：
   ```bash
   # 语法说明
   # wsl --port Ubuntu <WSL后续要放在哪个文件夹中> <镜像路径>
   wsl --import Ubuntu d:\WSL-Ubuntu-22.04 d:\image_ubuntu22.04.tar
   ```
7. 重新查看 WSL 状态：
   ```bash
   wsl -l -v
   ```
   ```
   NAME            STATE           VERSION
   * Ubuntu    Stopped         2
   ```
   此时，我们的 WSL 就已经移动完成了！

💡  **注意**：

1. 移动完成后不需要重新设置密码了
2. `image_ubuntu.tar` 这个文件可以删除掉了
3. `WSL-Ubuntu/` 这个文件夹就是 WSL2 的系统盘，不要删除！

## 2.2 修改默认账号

如果移动完毕后默认账号是 `root`，我们可以修改默认账号为我们自己的账号。

```bash
# 1. 编辑 wsl.conf 文件
vim /etc/wsl.conf

# 2. 添加下面内容
[user]
default=我们之前设置的账号名

# 3. 退出wsl
exit

# 4. 关闭 wsl
wsl --shutdown

# 5. 打开 wsl
wsl
```

此时 wsl 默认使用的是账户名就是我们之前的账号名了。

> 💡  不会使用 `vim` 则可以使用 `notepad.exe`。

# 3. WSL2 首次配置

## 3.1 更新软件包

安装完 WSL2 之后，我们就可以理解为它就是一个全新的系统，所以我们首先需要更新软件包：

```bash
sudo apt update
```

## 3.2 配置 Anaconda

### 3.2.1 安装 Anaconda

```bash
# 1. 先 cd 到根目录下
cd

# 2. 下载安装包：在此地址 https://www.anaconda.com/download/success 中找到安装包的链接
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

# 3. 安装 anaconda
bash Anaconda3-2024.02-1-Linux-x86_64.sh

# 4. 按照 anaconda 提示进行安装，默认安装到 /home/用户名/anaconda3
```

### 3.2.2 设置 Anaconda 环境变量

```bash
# 1. 打开系统环境变量文件
vim ~/.bashrc

# 2. 添加 Anaconda 环境变量
export PATH="/home/用户名/anaconda3/bin:$PATH"

# 3. （可选）设置 Anaconda 快捷键
alias act='conda activate'
alias deact='conda deactivate'

# 4. 更新环境变量
source ~/.bashrc

# 5. 验证是否添加完成
conda --version
```

得到下面的结果：

```
conda 24.1.2
```

此时，Anaconda 就已经安装好了！

💡  **Tips**：
1. 嫌弃 `wget` 下载慢的话，可以直接在 Windows 上下载 [Anaconda](https://www.anaconda.com/download/success)（注意是 Linux 版本，即 `64-Bit (x86) Installer (997.2M)`），之后 `cd` 到下载目录，安装即可🤗
2. 安装 Anaconda 时，协议太长了可以按 <kbd>q</kbd> 跳过（反正你也不看🤭）
3. Anaconda 安装可能会很慢，耐心一点
4. 在执行打开环境变量文件时，如果说没有找到 `~/.bashrc`，请直接 `cd` 到 `/` 后再执行
5. 在设置 Anaconda 环境变量路径时，`/home/WSL用户名/` 就是你 Anaconda 安装的位置。比如我直接安装在了 `/home` 下，所以就是 `export PATH="/home/anaconda3/bin:$PATH"`

## 3.3 创建 Anaconda 虚拟环境

因为 WSL2 是一个新系统，所以我们需要重新创建环境。

### 3.3.1 创建虚拟环境

```bash
# 创建虚拟环境
conda create -n 虚拟环境名称 python=3.10
```

创建过程示例如下：

```
(base) leovin@DESKTOP-XXXXX:~$ conda create -n dl python=3.10
Channels:
 - defaults
Platform: linux-64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /home/leovin/anaconda3/envs/dl

  added / updated specs:
    - python=3.10


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    bzip2-1.0.8                |       h5eee18b_5         262 KB
    ca-certificates-2024.3.11  |       h06a4308_0         127 KB
    pip-23.3.1                 |  py310h06a4308_0         2.7 MB
    python-3.10.14             |       h955ad1f_0        26.8 MB
    setuptools-68.2.2          |  py310h06a4308_0         957 KB
    tzdata-2024a               |       h04d1e81_0         116 KB
    wheel-0.41.2               |  py310h06a4308_0         109 KB
    xz-5.4.6                   |       h5eee18b_0         651 KB
    ------------------------------------------------------------
                                           Total:        31.7 MB

The following NEW packages will be INSTALLED:

  _libgcc_mutex      pkgs/main/linux-64::_libgcc_mutex-0.1-main
  _openmp_mutex      pkgs/main/linux-64::_openmp_mutex-5.1-1_gnu
  bzip2              pkgs/main/linux-64::bzip2-1.0.8-h5eee18b_5
  ca-certificates    pkgs/main/linux-64::ca-certificates-2024.3.11-h06a4308_0
  ld_impl_linux-64   pkgs/main/linux-64::ld_impl_linux-64-2.38-h1181459_1
  libffi             pkgs/main/linux-64::libffi-3.4.4-h6a678d5_0
  libgcc-ng          pkgs/main/linux-64::libgcc-ng-11.2.0-h1234567_1
  libgomp            pkgs/main/linux-64::libgomp-11.2.0-h1234567_1
  libstdcxx-ng       pkgs/main/linux-64::libstdcxx-ng-11.2.0-h1234567_1
  libuuid            pkgs/main/linux-64::libuuid-1.41.5-h5eee18b_0
  ncurses            pkgs/main/linux-64::ncurses-6.4-h6a678d5_0
  openssl            pkgs/main/linux-64::openssl-3.0.13-h7f8727e_0
  pip                pkgs/main/linux-64::pip-23.3.1-py310h06a4308_0
  python             pkgs/main/linux-64::python-3.10.14-h955ad1f_0
  readline           pkgs/main/linux-64::readline-8.2-h5eee18b_0
  setuptools         pkgs/main/linux-64::setuptools-68.2.2-py310h06a4308_0
  sqlite             pkgs/main/linux-64::sqlite-3.41.2-h5eee18b_0
  tk                 pkgs/main/linux-64::tk-8.6.12-h1ccaba5_0
  tzdata             pkgs/main/noarch::tzdata-2024a-h04d1e81_0
  wheel              pkgs/main/linux-64::wheel-0.41.2-py310h06a4308_0
  xz                 pkgs/main/linux-64::xz-5.4.6-h5eee18b_0
  zlib               pkgs/main/linux-64::zlib-1.2.13-h5eee18b_0


Proceed ([y]/n)? y


Downloading and Extracting Packages:

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate dl
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```




### 3.3.2  激活虚拟环境以及安装第三方库

```
# 1. 激活环境（💡  如果在 ~/.bashrc 中添加了快捷键，那么可以使用 act 代替 conda activate）
act 虚拟环境名称

# 2. 安装第三方库（-i https://pypi.tuna.tsinghua.edu.cn/simple 的目的是换源，可以加快下载速度）
pip install 第三方库的名称 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# 4. 安装 PyTorch

## 4.1 查看 CUDA 版本

```bash
# 查看显卡状态
nvidia-smi
```

示例如下：

```
(dl) leovin@DESKTOP-XXXX:~$ nvidia-smi
Sat Apr 27 23:16:39 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.76.01              Driver Version: 552.22         CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3070        On  |   00000000:01:00.0  On |                  N/A |
|  0%   47C    P8             15W /  240W |    1258MiB /   8192MiB |      8%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## 4.2 在 PyTorch 官网找到相应的安装命令

💡  **Tips**：
- [PyTorch 官网链接](https://pytorch.org/get-started/locally/)
- 请安装对应 CUDA 版本的 PyTorch，如果 CUDA 版本大于 PyTorch 支持的最大版本，则选择最大版本。比如我的 CUDA 版本是 12.4，但截止 2024/04/27，PyTorch 支持的最大 CUDA 版本为 12.1，那么则选择 12.1 进行安装
- 如果没有 GPU，则安装 CPU 版本

```bash
# 这里添加 -i 是为了加速其他第三方包的下载速度
pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
```



## 4.3 测试 PyTorch 是否可用

```bash
# 1. 打开 Python
python

# 2. 导入 PyTorch
import torch

# 3. 查看 PyTorch 版本
print(torch.__version__)

# 4. 查看 CUDA 是否可用（💡  如果安装的是 CPU 版本则不用运行）
print(f"{torch.cuda.is_available() = }")
```

示例结果如下：

```bash
(dl) leovin@DESKTOP-XXXX:~$ python
Python 3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> print(torch.__version__)
2.3.0+cu121
>>> print(f"{torch.cuda.is_available() = }")
torch.cuda.is_available() = True
```

# 4. 使用 VSCode 打开 WSL2

## 4.1 方式1

直接在我们想要打开的文件夹下，<kbd>Shift + 右键</kbd>，选择 <kbd>在此处打开 Linux shell(L)</kbd>，之后在打开的终端输入 `code ./` 即可。或者直接在 WSL2 终端中输出 `code ./`，之后手动再次选择文件夹也可以。

<div align=center>
    <img src=https://img-blog.csdnimg.cn/24588ba7d7b6479dbcce9f8968b2f7ed.png
    width=35%>
</div>


```bash
# 使用 VSCode 打开当前路径的📂文件夹
code ./
```

```
Installing VS Code Server for x64 (f1b07bd25dfad64b0167beb15359ae573aecd2cc)
Downloading: 100%
Unpacking: 100%
Unpacked 1608 files and folders to /root/.vscode-server/bin/xxxxxxxxxxxxxxxxxxxxx
```

这里是提示我们要安装 VSCode，等待安装完毕即可。

## 4.2 方式2

当我们安装好 WSL2 后，可以在 VSCode 中搜索 [WSL 插件](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl)，安装完毕后即可在 VSCode 中远程链接 WSL2 了（和 SSH 远程链接服务器差不多）。

# 5. WSL2 和 Windows 共享 Proxy

1. 打开 <kbd>Allow LAN（允许局域网）</kbd>
2. 打开环境变量
   ```bash
   cd
   notepad.exe .bashrc
   ```
3. 添加语句：
   ```bash
   hostip=$(cat /etc/resolv.conf |grep -oP '(?<=nameserver\ ).*')
   export https_proxy="http://${hostip}:7890"
   export http_proxy="http://${hostip}:7890"
   export all_proxy="socks5://${hostip}:7890"
   ```
   此处需要注意的是，`7890` 是你设置的端口号，可以在 Proxy 软件中的 `port` 中设置
4. 更新环境变量
   ```bash
   source .bashrc
   ```
5. 可以通过 `ping` 命令来进行测试

<kbd>Note</kbd>
   1. 如果之后失效了，关闭 <kbd>Allow LAN（允许局域网）</kbd> 再打开就可以了。

# 知识来源

1. [如何使用 WSL 在 Windows 上安装 Linux](https://learn.microsoft.com/zh-cn/windows/wsl/install)
2. [WSL2 修改安装目录](https://www.bilibili.com/read/cv17865605/)
