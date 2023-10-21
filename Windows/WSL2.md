
# 1. WSL2 安装

1. 以管理员身份打开 PowerShell（“开始”菜单 >“PowerShell” >单击右键 >“以管理员身份运行”），然后输入以下命令：

    ```bash
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    ```

2. 以管理员身份打开 PowerShell 并运行：
   ```bash
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```

3. :exclamation:重新启动计算机(必须)
4. 下载内核并安装: [适用于 x64 计算机的 WSL2 Linux 内核更新包](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi)
5. 打开 PowerShell，然后在安装新的 Linux 发行版时运行以下命令，将 WSL 2 设置为默认版本
   ```bash
   wsl --set-default-version 2
   ```
6. 打开 Microsoft Store，并下载喜欢的 Linux 分发版。
   + [Ubuntu 18.04 LTS](https://www.microsoft.com/store/apps/9N9TNGVNDL3Q)
   + [Ubuntu 20.04 LTS](https://www.microsoft.com/store/apps/9n6svws3rx71)
   + [Ubuntu 22.04 LTS](https://www.microsoft.com/store/apps/9PN20MSR04DW)
   + [Debian GNU/Linux](https://www.microsoft.com/store/apps/9MSVKQC78PK6)
   + [Fedora Remix for WSL](https://www.microsoft.com/store/apps/9n6gdm4k2hnc)
7. 安装完成后在 Microsoft Store 打开安装的系统，首次启动新安装的 Linux 分发版时，将打开一个控制台窗口，系统会要求你等待一分钟或两分钟，以便文件解压缩并存储到电脑上。 未来的所有启动时间应不到一秒。
   <div align=center>
    <img src=./imgs_markdown/2023-10-19-10-26-20.png
    width=100%>
   </div>
8. 然后，需要为 Linux 分发版创建用户帐户和密码。
9. 之后在 cmd 或 Powershell 中输入 wsl 即可打开 WSL（直接 <kbd>Windows + R</kbd>，输入 `wsl` 即可运行 WSL2）。

# 2. 更改 WSL 所在路径

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
    --export Ubuntu-20.04 f:\image_ubuntu20.04.tar
   ```
   之后会在 F 盘中有一个大小为 1.2G 的 `image_ubuntu20.04.tar` 文件
4. 移除之前注册的 WSL：
   ```bash
   wsl --unregister Ubuntu-20.04
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
   wsl --import Ubuntu-20.04 f:\WSL-Ubuntu-20.04 f:\image_ubuntu20.04.tar
   ```
7. 重新查看 WSL 状态：
   ```bash
   wsl -l -v
   ```
   ```
   NAME            STATE           VERSION
   * Ubuntu-20.04    Stopped         2
   ```
   此时，我们的 WSL 就已经移动完成了！

<kbd>注意</kbd>：
   1. 移动完成后不需要重新设置密码了
   2. `image_ubuntu20.04.tar` 这个文件可以删除掉了
   3. `WSL-Ubuntu-20.04/` 这个文件夹就是 WSL2 的系统盘，不要删除！

# 3. WSL2 首次配置

## 3.1 更新软件包

安装完 WSL2 之后，我们就可以理解为它就是一个全新的系统，所以我们首先需要更新软件包：

```bash
sudo apt update
```

## 3.2 安装 Anaconda

1. 下载安装包
   ```bash
   cd
   wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
   ```
2. 安装
   ```bash
   sh Anaconda3-2023.09-0-Linux-x86_64.sh
   ```
3. 打开环境变量
   ```bash
   cd
   notepad.exe ~/.bashrc
   ```

4. 设置环境变量
   ```bash
   # 添加 Anaconda 环境变量
   export PATH="/home/WSL用户名/anaconda3/bin:$PATH"
   ```
5. 设置快捷键
   ```bash
   # 设置快捷键(可选)
   alias act='conda activate'
   alias deact='conda deactivate'
   ```
6. 更新环境变量
   ```bash
   source ~/.bashrc
   ```
7. 此时在 bash 中输入：
   ```bash
   conda --version
   ```

   得到下面的结果：

   ```
   conda 23.7.4
   ```
   此时，Anaconda 就已经安装好了！

<kbd>Tips</kbd>:

   1. 嫌弃 `wget` 下载慢的话🤪，可以直接在 Windows 上下载 [Anaconda](https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh)，之后 `cd` 到下载目录，安装即可🤗
   2. 安装 Anaconda 时，协议太长了可以按 <kbd>q</kbd> 跳过（反正你也不看🤭）
   3. Anaconda 安装可能会很慢，耐心一点🫡
   4. 在执行 <kbd>step 3</kbd> 时，如果说没有找到 `~/.bashrc`，请直接 `cd` 到 `root` 后再执行
   5. 在执行 <kbd>step 4</kbd> 时，`/home/WSL用户名/` 就是你 Anaconda 安装的位置。比如我直接安装在了 `root` 下，所以就是 `export PATH="/root/anaconda3/bin:$PATH"`

## 3.3 创建 Anaconda 虚拟环境

因为 WSL2 是一个新系统，所以我们需要重新创建环境。

1. 创建环境
   ```bash
   conda create -n 虚拟环境名称 python=3.8
   ```

   ```
   Downloading and Extracting Packages

   Preparing transaction: done
   Verifying transaction: done
   Executing transaction: done
   #
   # To activate this environment, use
   #
   #     $ conda activate learning
   #
   # To deactivate an active environment, use
   #
   #     $ conda deactivate
   ```
   此时，虚拟环境安装完毕
2. 初始化 conda 环境
   ```bash
   conda init
   ```
   之后退出这个 bash，重新打开一个
3. 激活虚拟环境
   ```bash
   act 虚拟环境名称
   ```
4. 安装必要的库
   ```bash
   pip install 库的名称
   ```

# 4. 安装 PyTorch

1. 先查看 CUDA 版本：
   ```bash
   nvidia-smi
   ```

   ```
   +-----------------------------------------------------------------------------+
   | NVIDIA-SMI 510.60.02    Driver Version: 512.15       CUDA Version: 11.6     |
   |-------------------------------+----------------------+----------------------+
   | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
   |                               |                      |               MIG M. |
   |===============================+======================+======================|
   |   0  NVIDIA GeForce ...  On   | 00000000:01:00.0  On |                  N/A |
   |  0%   35C    P8    13W / 240W |    719MiB /  8192MiB |      5%      Default |
   |                               |                      |                  N/A |
   +-------------------------------+----------------------+----------------------+

   +-----------------------------------------------------------------------------+
   | Processes:                                                                  |
   |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
   |        ID   ID                                                   Usage      |
   |=============================================================================|
   |  No running processes found                                                 |
   +-----------------------------------------------------------------------------+   
   ```
2. 在 [PyTorch 官网](https://pytorch.org/get-started/locally/) 上找到对应的安装命令：
   ```bash
   pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
   ```
   ```
   Successfully installed certifi-2023.7.22 charset-normalizer-3.3.0 idna-3.4 pillow-10.1.0 requests-2.31.0 torch-1.13.1+cu116 torchaudio-0.13.1+cu116 torchvision-0.14.1+cu116 typing-extensions-4.8.0 urllib3-2.0.7
   ```
3. 测试 PyTorch 是否可用：
   ```bash
   python
   ```

   ```python
   >>> import torch
   >>> dummpy_tensor = torch.ones((1, 2, 3))
   >>> dummpy_tensor.cuda()
   tensor([[[1., 1., 1.],
            [1., 1., 1.]]], device='cuda:0')
   ```
   现在，我们的 CUDA 版本的 PyTorch 就可以用了！

# 4. 使用 VSCode 打开 WSL2

## 4.1 方式1

直接在我们想要打开的文件夹下，<kbd>Shift + 右键</kbd>，选择 <kbd>在此处打开 Linux shell(L)</kbd>，之后在打开的终端输入 `code ./` 即可。或者直接在 WSL2 终端中输出 `code ./`，之后手动再次选择文件夹也可以。

<div align=center>
    <img src=./imgs_markdown/2023-10-21-21-30-36.png
    width=35%>
</div>

```bash
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

1. [旧版 WSL 的手动安装步骤](https://learn.microsoft.com/zh-cn/windows/wsl/install-manual)
2. [WSL2 修改安装目录](https://www.bilibili.com/read/cv17865605/)