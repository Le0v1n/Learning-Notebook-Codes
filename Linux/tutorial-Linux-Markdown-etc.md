# 1. Linux命令
### 1.1 常用
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

### 1.2 一般

1. 更新软件包列表：`sudo apt update`
2. 在 WSL 中用记事本打开文档: `notepad.exe 文件路径`
3. screen:
   1. 创建 session: `screen -R session_name`
   2. 进入 session: `screen -r session_name`
4. 从其他服务器复制文件到另一台服务器：
   1. 先进入 [文件所在服务器]
   2. `sudo scp -p -r 端口号 要复制的文件夹 目标服务器名称@IP地址:目标文件夹`

### 1.3 罕见
1. 修改 sudo 密码: `sudo passwd`

# 2. Markdown

## 2.1 Markdown 基础语法

<div align=center>

| 效果 | 语法 | 说明 |
| :- | :- | :-: |
||`#`|标题|
| _斜体_ |`_斜体_` 或 `*斜体*`| `-` 和 `*` 都可以 |
| **粗体** |`**粗体**`|  |
| ***粗斜体*** |`***粗斜体***`| 粗体 + 斜体 |
|  |`---`| 分割线 |
| ~~删除线~~ |`~~删除线~~`|  |
| <u>下划线</u> |`~~下划线~~`|  |
| 脚注[^脚注的名字] | 添加脚注→ `脚注[^脚注的名字]`<br>写脚注→ `[^脚注的名字]: 脚注的内容`| 记着写脚注的具体内容 |
| 1. 有序列表 |`1. 有序列表`|  |
| + 无序列表 |`+ 无序列表`| `*` `-` `+` 都可以 |
| - [ ] 你好 |`- [ ] 你好`| 待办事项 |
| - [x] 你好 |`- [x] 你好`| 已办事项 |
|  |`> 区块内容`| 区块 |
| `代码` |` `` `|  |
| ```代码块``` |` ```代码块``` `|  |
| ```代码块``` |` ```代码块``` `| 注意写代码语言 |
| [链接地址](https://blog.csdn.net/weixin_44878336) |`[要显示的内容](具体网址)`|  |

</div>

[^脚注的名字]: 这是一个演示的脚注（脚注的内容）



## 2.2 Markdown 高级语法

1. 换行符: `<br> 内容 </br>`
2. 居中符: `<center> 内容 </center>`
3. 加粗符: `<b> 内容 </b>`
4. 按键效果：`<kbd> 内容 </kbd>` —— <kbd> 内容 </kbd>
5. <font color='red'>换颜色</font>: `<font color='red'></font>`
6. 调整字体大小: `<font size=12 color='red'> 内容 </font>`
7. 图片居中
   ```markdown
   <div>
      <img src=图片链接
      width=100%>
   </div>
   ```
8. 图片并排显示
   ```markdown
   <center class="half">
      <img src="img1.jpg" width="270"/>
      <img src="img2.jpg" width="270"/>
   </center>
   ```
9. 折叠块
   ```markdown
   <details>

      <summary>展开/折叠</summary>

      具体内容...

   </details>
   ```
10. mermaid 画图
    1.  `graph TB;`
    2.  `graph LR;`
    ```markdown
      ```mermaid
      graph TB;
         A-->B;
         A-->C;
         B-->D;
      ```
    ```
11. 插入视频
   ```css
   <video id="video" controls="" preload="none"> 
      <source id="mp4" src="本地视频路径.mp4"
      type="video/mp4"> 
   </video>
   ```
12. 表格
    1. `-:` 设置内容或标题栏右对齐
    2. `:-` 设置内容或标题栏左对齐
    3. `:-:` 设置内容或标题栏居中对齐


## 2.3 LaTex 公式

### 2.3.1 语法

以下是包含 LaTeX 代码和符号说明的数学排版表格：

<div align=center>

| 数学符号 | LaTeX 代码 | 符号说明 |
| :-: | :-: | :-: |
| $A \ B$ | `$A \ B$` | 空格 |
| $A \quad B$ | `$A \quad B$` | 四个空格 |
| $A \\ B$ | `$A \\ B$` | 换行 |
| $\{a, b \}$ | `\\{a, b \\}` | 转义字符 `\` |
| $\hat{x}$ | `$\\hat{x}$` | 帽子 |
| $\bar{x}$ | `$\\bar{x}$` | 短横线 |
| $\overline{xyz}$ | `$\\overline{xyz}$` | 长横线 |
| $\underline{xyz}$ | `$\\underline{xyz}$` | 长下划线 |
| $\dot{x}$ | `$\\dot{x}$` | 一个点 |
| $\ddot{x}$ | `$\\ddot{x}$` | 两个点 |
| $\vec{x}$ | `$\\vec{x}$` | 矢量 |
| $\overrightarrow{x}$ | `$\\overrightarrow{x}$` | 长矢量 |
| $\left( abc \right)$ | `$\\left( abc \\right)$` | 长括小括号 |
| $\left[ abc \right]$ | `$\\left[ abc \\right]$` | 长括中括号 |
| $\underset{A}{B}$ | `$\underset{A}{B}$` | 在下方写 |
| $\overset{A}{B}$ | `$\overset{A}{B}$` | 在上方写 |

</div>

### 2.3.2 字体

<div align=center>

| 数学符号 | LaTeX 代码 | 符号说明 |
| :-: | :- | :- |
| $\rm{Hello}$ | `$\rm{Hello}$` | 非斜体罗马字体 |
| $\mathit{Hello}$ | `$\mathit{Hello}$` | 斜体字体 |
| $\mathsf{Hello}$ | `$\mathsf{Hello}$` | Sans serif 字体 |
| $\mathtt{Hello}$ | `$\mathtt{Hello}$` | Typerwriter 字体 |
| $\mathcal{Hello}$ | `$\mathcal{Hello}$` | Calligraphic 字体 |
| $\mathbb{Hello}$ | `$\mathbb{Hello}$` | Blackboard bold 字体 |
| $\boldsymbol{Hello}$ | `$\boldsymbol{Hello}$` | Boldsymbol bold 字体 |

</div>

### 2.3.3 矩阵、对齐、分段函数

1. 【矩阵】$\left[\begin{matrix}a & b \cr c & d\end{matrix}\right]$
   ```
   \left[\begin{matrix}
      a & b \cr 
      c & d
   \end{matrix}\right]
   ```
2. 【矩阵】$\left\lgroup\begin{matrix}a & b \cr c & d\end{matrix}\right\rgroup$
   ```
   \left\lgroup\begin{matrix}
      a & b \cr
       c & d
   \end{matrix}\right\rgroup
   ```
3. 【对齐】
   $$
      \begin{aligned}
      a_1 &= b_1 + c_1 \\
      a_2 &= b_2 + c_2 + d_2 \\
      a_3 &= b_3 + c_3
      \end{aligned}
   $$

   ```
   \begin{aligned}
   a_1 &= b_1 + c_1 \\
   a_2 &= b_2 + c_2 + d_2 \\
   a_3 &= b_3 + c_3
   \end{aligned}
   ```
4. 【分段函数】语法中的 `\\` 等价于 `\cr`，表示换行。
   $$
   sign(x) = 
   \begin{cases}
      1, & x > 0 \\ 
      0, & x = 0 \cr 
      -1, & x < 0
   \end{cases}
   $$

   ```
   sign(x) = 
   \begin{cases}
      1, & x > 0 \\ 
      0, & x = 0 \cr 
      -1, & x < 0
   \end{cases}
   ```

### 2.3.4 希腊字母

<div align=center>

| 数学符号 | LaTeX 代码 | 对应大写字母 | LaTeX 代码 |
| :-: | :- | :-: | :- |
| $\alpha$ | `$\alpha$` | $\Gamma$ | `$\Gamma$` |
| $\beta$ | `$\beta$` | $\Delta$ | `$\Delta$` |
| $\gamma$ | `$\gamma$` | $\Theta$ | `$\Theta$` |
| $\delta$ | `$\delta$` | $\Delta$ | `$\Delta$` |
| $\epsilon$ | `$\epsilon$` | | |
| $\varepsilon$ | `$\varepsilon$` | | |
| $\zeta$ | `$\zeta$` | | |
| $\eta$ | `$\eta$` | | |
| $\theta$ | `$\theta$` | $\Theta$ | `$\Theta$` |
| $\vartheta$ | `$\vartheta$` | $\varTheta$ | `$\varTheta$` |
| $\iota$ | `$\iota$` | | |
| $\kappa$ | `$\kappa$` | | |
| $\lambda$ | `$\lambda$` | $\Lambda$ | `$\Lambda$` |
| $\mu$ | `$\mu$` | | |
| $\nu$ | `$\nu$` | | |
| $\xi$ | `$\xi$` | $\Xi$ | `$\Xi$` |
| $\pi$ | `$\pi$` | $\Pi$ | `$\Pi$` |
| $\varpi$ | `$\varpi$` | $\varPi$ | `$\varPi$` |
| $\rho$ | `$\rho$` | | |
| $\varrho$ | `$\varrho$` | | |
| $\sigma$ | `$\sigma$` | $\Sigma$ | `$\Sigma$` |
| $\varsigma$ | `$\varsigma$` | $\varSigma$ | `$\varSigma$` |
| $\tau$ | `$\tau$` | | |
| $\upsilon$ | `$\upsilon$` | $\Upsilon$ | `$\Upsilon$` |
| $\phi$ | `$\phi$` | $\Phi$ | `$\Phi$` |
| $\varphi$ | `$\varphi$` | $\varPhi$ | `$\varPhi$` |
| $\chi$ | `$\chi$` | | |
| $\psi$ | `$\psi$` | $\Psi$ | `$\Psi$` |
| $\omega$ | `$\omega$` | $\Omega$ | `$\Omega$` |

</div>

### 2.3.5 运算符

<div align=center>

| 数学符号 | LaTeX 代码 | 说明 |
| :-: | :- | :-: |
| $\ll$ | `$\ll$` | 远小于 |
| $\gg$ | `$\gg$` | 远大于 |
| $\approx$ | `$\approx$` | 约等于 |
| $\sim$ | `$\sim$` | 相似 |
| $\ne$ | `$\ne$` | 不等于 |
| $\in$ | `$\in$` | 属于 |
| $\cup$ | `$\cup$` | 交 |
| $\cap$ | `$\cap$` | 并 |
| $\pm$ | `$\pm$` | 加减 (plusminus) |
| $\div$ | `$\div$` | 除法 |
| $\cdot$ | `$\cdot$` | 点乘 |
| $\odot$ | `$\odot$` | 圈点乘 |
| $\oplus$ | `$\oplus$` | 圈加 |
| $\otimes$ | `$\otimes$` | 圈乘 |
| $\prod$ | `$\prod$` | 连乘 |
| $\int$ | `$\int$` | 积分 |
| $\partial$ | `$\partial$` | 偏导 |

</div>

### 2.3.6 其他符号

<div align=center>

| 数学符号 | LaTeX 代码 | 说明 |
| :-: | :- | :- |
| $\dots$ | `$\dots$` | 省略号 |
| $\cdots$ | `$\cdots$` | 居中省略号 |
| $\Re$ | `$\Re$` | 实部 |
| $\nabla$ | `$\nabla$` | 梯度符号 |
| $\triangle$ | `$\triangle$` | 三角形 |
| $\angle$ | `$\angle$` | 角度符号 |
| $\infty$ | `$\infty$` | 无穷大 |
| $\dag$ | `$\dag$` | 剪影标记 |
| $\ddag$ | `$\ddag$` | 双剪影标记 |
| $\S$ | `$\S$` | 资料标记 |
| $\because$ | `$\because$` | 因为 |
| $\therefore$ | `$\therefore$` | 所以 |
| $\leftrightarrow$ | `$\leftrightarrow$` | 左右箭头 |
| $\Leftrightarrow$ | `$\Leftrightarrow$` | 左右双箭头 |
| $\nleftrightarrow$ | `$\nleftrightarrow$` | 非左右箭头 |
| $\nLeftrightarrow$ | `$\nLeftrightarrow$` | 非左右双箭头 |
| $\varnothing$ | `$\varnothing$` | 空集符号 |

</div>

# 3. WSL2 的安装

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

# 4. Linux 安装 Anaconda 以及 设置环境变量

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
# 5. 配置 Jupyter Notebook

## 5.1 将 conda 的环境添加到 notebook 中

```shell
conda activate 虚拟环境名
conda install ipykernel
python -m ipykernel install --user --name 虚拟环境名 --display-name "自定义名字"
jupyter kernelspec list   #查看当前notebook中所具有的kernel
```

在该虚拟环境中还需要重新安装jupyter notebook

```bash
pip install jupyter notebook
```

## 5.2 代码自动填充 Auto-fill

```bash
pip install jupyter_contrib_nbextensions

jupyter contrib nbextension install --user

pip install --user jupyter_nbextensions_configurator 

jupyter nbextensions_configurator enable --user

jupyter nbextension enable
```

# 6. KMS 主机地址

```https
kms.loli.beer
kms.loli.best
kms.03k.org
kms.cary.tech
kms.mc06.net
```

# 参考

1. [如何使用jupyter编写数学公式(译)](https://www.jianshu.com/p/93ccc63e5a1b)
2. [旧版 WSL 的手动安装步骤](https://learn.microsoft.com/zh-cn/windows/wsl/install-manual)