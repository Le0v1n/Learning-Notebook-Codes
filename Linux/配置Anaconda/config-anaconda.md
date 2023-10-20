
1. 下载安装包
   ```bash
   wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
   ```
2. 安装
   ```bash
   sh Anaconda3-2023.09-0-Linux-x86_64.sh
   ```
3. 设置环境变量和设置快捷键
   ```bash
   gedit(notepad.exe) ~/.bashrc
   ```
   添加一行:
   ```bash
   # 添加 Anaconda 环境变量
   export PATH="/home/用户名/anaconda3/bin:$PATH"

   # 设置快捷键(可选)
   alias act='conda activate'
   alias deact='conda deactivate'
   ```