
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