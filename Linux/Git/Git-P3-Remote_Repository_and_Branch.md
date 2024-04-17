
# 8. github

## 8.1 创建一个远程仓库

我们可以在 [GitHub](https://github.com/) 上创建一个代码仓库。

## 8.2 .gitignore

`.gitignore` 表明 Git 不要管理的文件，因为程序在运行的时候会生成很多 `cache` 文件、代码生成的结果等，而这些文件并不是必须的，所以我们希望 `git` 不要管理这些文件。

## 8.3 添加 SSH 账号

1. 点击账户头像后的下拉三角，选择 `settings`。如果某台机器需要与 github 上的仓库交互，那么就要把这台机器的 ssh 公钥添加到这个 github 账户上
2. 在 Ubuntu 的命令行中，回到用户的主目录下，编辑文件 `.gitconfig`，修改某台机器的 git 配置。
3. 修改为注册 github 时的邮箱，填写用户名。
4. 使用如下命令生成 ssh 密钥。

```bash
ssh-keygen -t rsa -C "email"
```

<a></a>
<div align=center>
    <img src=https://img-blog.csdnimg.cn/94be78ebfdf94a1fb92e72ca29fde84b.png
    width=50%>
    <center></center>
</div></br>

其中，`id_rsa` 为私钥，我们自己保留；`id_rsa.pub` 为公钥

5. 使用 `cat id_rsa.pub` 查看公钥的内容
6. 回到浏览器中，填写标题，粘贴公钥

## 8.1 克隆项目

1. 在浏览器中点击进入 github 首页，再进入项目仓库的页面
2. 复制 git 地址
3. 在命令行中复制仓库中的内容

<a></a>
<div align=center>
    <img src=https://img-blog.csdnimg.cn/4502500b3fad4db4acb846a39082873b.png
    width=90%>
    <center></center>
</div></br>

## 8.2 上传分支

通常我们不会在 main 分支上进行开发，一般会创建自己的分支进行开发，最后整合到 main 分支上。

> ⚠️  OBS：现在 github 默认的分支名称为 main 分支而非 master 分支。

1. 项目克隆到本地之后，执行如下命令创建分支 dev

```bash
# 创建 dev 分支
git checkout -b dev
```

2. 创建一个 `views.py` 并提交一个版本。

```bash
# 创建文件
touch views.py

# add
git add views.py

# commit
git commit -m ""
```

3. 推送分支，就是把该分支上的所有本地提交推送到远程库，推送时要指定本地分支，这样，Git 就会把该分支推送到远程库对应的远程分支上：

```bash
git push origin <branch name>
```

> origin: 英`[ˈɒrɪdʒɪn]` 美`[ˈɔːrɪdʒɪn]` n.起源;出身;源头;起因;身世;

<a></a>
<div align=center>
    <img src=https://img-blog.csdnimg.cn/eb1ea5004f6747d8909fd7628651264f.png
    width=90%>
    <center></center>
</div></br>

<a></a>
<div align=center>
    <img src=https://img-blog.csdnimg.cn/93bf43facf6a4c53b13c76ba72fd25b5.png
    width=90%>
    <center></center>
</div></br>

<a></a>
<div align=center>
    <img src=https://img-blog.csdnimg.cn/57a92ce59846422385fceac07c1a13ed.png
    width=90%>
    <center></center>
</div></br>

## 8.3 本地分支跟踪远程分支

设置本地分支跟踪远程分支有以下几个重要原因：

1. **同步代码**：跟踪远程分支可以使我们轻松地将本地分支与远程分支保持同步。当远程分支有新的提交时，我们可以通过拉取（pull）操作将最新的更改合并到本地分支，确保我们的代码是最新的。

2. **管理追踪关系**：跟踪远程分支可以将本地分支与特定的远程分支建立关联，使我们更清楚地知道本地分支与哪个远程分支相关联。这对于团队协作和多个开发者同时工作在同一个项目上非常有帮助。

3. **简化操作**：通过设置本地分支跟踪远程分支，我们可以使用更简洁的命令来执行常见的操作，如推送（push）和拉取（pull）。而不需要每次都指定远程分支的名称。

总而言之，设置本地分支跟踪远程分支可以提供更好的代码管理和协作体验，确保我们的代码与团队的工作保持同步，并简化常见操作的执行。

```bash
git branch <tracking branch name> --set-upstream-to=origin/<tracked branch name>
```

<a></a>
<div align=center>
    <img src=https://img-blog.csdnimg.cn/32338720ff5c4566b0e51bbe35c88efa.png
    width=90%>
    <center></center>
</div></br>

<a></a>
<div align=center>
    <img src=https://img-blog.csdnimg.cn/51616b880e6e4e6086cb670c00da37ef.png
    width=80%>
    <center></center>
</div></br>

我们在 local branch 修改 `views.py` 并提交：

```bash
# 编辑文件
gedit views.py

# add
git add.

# commit
git commit -m "add new functions -> login"
```

<a></a>
<div align=center>
    <img src=https://img-blog.csdnimg.cn/3bbc97df34404676a3e8a2d68deb2620.png
    width=90%>
    <center></center>
</div></br>

我们查看一下 git status：

```bash
git status
```

<a></a>
<div align=center>
    <img src=https://img-blog.csdnimg.cn/23b472a099564cf7ba381abfef886417.png
    width=90%>
    <center></center>
</div></br>

💡  OBS：
+ 当我们的本地 branch 跟踪了 remote branch，我们 push 的时候就不用指定 `origin` 了。

```bash
# 情况1：如果本地分支没有跟踪远程分支
git push origin <branch name>

# 情况2：如果本地分支跟踪了远程分支
git push

# 再次查看 status
git status
```

<a></a>
<div align=center>
    <img src=https://img-blog.csdnimg.cn/f0780fd7d3394e4c98899ea4a4cabb39.png
    width=90%>
    <center></center>
</div></br>

## 8.4 从远程分支拉取代码

```bash
git pull origin <pulled branch name>
```

<a></a>
<div align=center>
    <img src=https://img-blog.csdnimg.cn/7f77cf072d6546edbaadbd1ab75ba8c7.png
    width=90%>
    <center></center>
</div></br>

使用上述命令会把远程分支 dev 上的代码下载并合并到本地所在分支。

# 9. 在真实场景中使用 Git

## 9.1 项目经理干的活

1. 项目经理搭建项目的框架
2. 搭建完项目框架之后，项目经理把项目框架代码放到服务器(自己搭建或者使用 github)

## 9.2 打工人干的活

1. 在自己的电脑上，生成 ssh 公钥，然后把公钥给项目经理，项目经理把它添加的服务器上面。
2. 项目经理会给每个组员的项目代码的克隆地址，组员把代码下载到自己的电脑上。
3. 创建本地的分支 dev,在 dev 分支中进行每天的开发。
4. 每一个员工开发完自己的代码之后，都需要将代码发布到远程的 dev 分支上。

> 💡  OBS：不是说每次写完代码都要 push 到 remote branch，而是每次写完代码先 push 到自己的 local branch 中，最后经过一系列测试后我们才 push 到 remote branch :joy:

# 10. 创建本地仓库

## 10.1 方法 1

```bash
# 1. 创建仓库的本地文件夹
mkdir 本地仓库的文件夹名称

# 2. 进入该文件夹
cd 本地仓库的文件夹名称

# 3. Git 初始化（这个文件夹就是我们的本地仓库了）
git init

# 4. 链接到远程仓库
git remote add origin 远程仓库的https地址

# 5. 从远程仓库拉取代码到本地的文件夹（本地仓库）
git pull origin main  # 我们可以自定义文件夹的名称，这里是 "main"

# 6. 这个操作需要我们的 GitHub 账户和密码

# 7. 我们可以使用 ls 命令在本地仓库中查找与远程仓库相同的文件
ls
```

## 10.2 方法 2：克隆一个远程仓库使其变为本地仓库

```bash
# 1. 创建仓库的本地文件夹
mkdir 本地仓库的文件夹名称

# 2. 进入该文件夹
cd 本地仓库的文件夹名称

# 3. 克隆一个远程仓库到本地
git clone 远程仓库的https地址
```