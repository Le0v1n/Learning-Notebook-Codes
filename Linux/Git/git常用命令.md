# 1. 账号相关

## 1. 生成SSH-Key

```bash
# 查看ssh目录是否存在
cd ~/.ssh

# 如果不存在则需要创建（将 "xxx@xxx.com" 替换为你自己GitHub的邮箱地址）
ssh-keygen -t rsa -C "xxx@xxx.com"

# 查看SSH-Key公钥信息
cd ~/.ssh
cat id_rsa.pub

# 复制公钥信息

# 在GitHub中添加公钥，URL为：https://github.com/settings/keys
```

## 2. 克隆仓库

```bash
git clone <ssh链接>
```

## 3. 创建账户

在`git commit`的时候可能会碰到问题 “please tell me who you are”，输入下面的命令：

```bash
# "you@example.com"替换为你的GitHub邮箱地址
git config --global user.email "you@example.com"

# "Your Name"替换为你的GitHub名称
git config --global user.name "Your Name"
```

## 4. git push需要输入账号密码

运行以下命令将远程仓库地址更新为 SSH 地址：

```bash
git remote set-url origin git@github.com:<用户名>/<仓库名>.git
```

# 2. 命令

## 1. branch分支相关

```bash
# 列出所有分支
git branch

# 创建新的分支
git branch <branch_name>

# 删除某个分支
git branch -d <branch_name>

# 切换到某个分支
git checkout <branch_name>
```

## 2. 提交代码

```bash
# 将文件或文件夹添加到暂存区
git add filepath/dirpath
git add .

# 将修改放到本地仓库
git commit -m "your useful commit message"

# 提交代码
git push
```