
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
9. 之后在 cmd 或 Powershell 中输入 wsl 即可打开 WSL。

# 知识来源

1. [旧版 WSL 的手动安装步骤](https://learn.microsoft.com/zh-cn/windows/wsl/install-manual)