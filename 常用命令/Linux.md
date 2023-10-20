### 1. 常用
1. 查看文件夹下文件数量: `ls -l | wc -l`
2. 7zip:
   1. 解压：`7z x compressed_file.7z -o/path/to/destination`  # 注意-o和目标路径是连起来的，没有空格
   2. 压缩：`7z a compressed_file.zip destination_path`
3. conda
   1. 查看 conda 拥有的环境: `conda env list`
   2. 创建 conda 环境: `conda create -n 环境名 python=3.8`
   3. 克隆 conda 环境: `conda create -n 环境名 --clone 要克隆哪个环境`
   4. 删除 conda 环境: `conda remove -n 环境名 --all`
   5. 

### 2. 一般

1. 更新软件包列表：`sudo apt update`
2. 在 WSL 中用记事本打开文档: `notepad.exe 文件路径`
3. screen:
   1. 创建 session: `screen -R session_name`
   2. 进入 session: `screen -r session_name`
4. 从其他服务器复制文件到另一台服务器：
   1. 先进入 [文件所在服务器]
   2. `sudo scp -p -r 端口号 要复制的文件夹 目标服务器名称@IP地址:目标文件夹`

### 3. 罕见
1. 修改 sudo 密码: `sudo passwd`