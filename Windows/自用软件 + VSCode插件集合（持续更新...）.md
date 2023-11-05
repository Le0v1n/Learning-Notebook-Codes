# 1. 软件名称及其链接

1. <kbd>IDE</kbd> ：[VSCode](https://code.visualstudio.com/)
2. <kbd>解压软件</kbd>：[7-zip](https://www.7-zip.org/)
3. <kbd>卸载软件</kbd>：[Geek Uninstaller](https://geekuninstaller.com/)
4. <kbd>录屏</kbd>：[OBS Studio](https://obsproject.com/)
5. <kbd>终端</kbd>：[Git Bash](https://blog.csdn.net/weixin_44878336/article/details/132698736)
6. <kbd>输入法</kbd>：[搜狗输入法](https://pinyin.sogou.com/mac/)
7. <kbd>中英文浮窗显示</kbd>：[ImTip ( 通用输入法状态跟踪提示 )](https://github.com/aardio/ImTip)
8. <kbd>GIF录制</kbd>：[GifCam](https://gifcam.en.softonic.com/) | :star:[Gif123](https://github.com/aardio/Gif123)
9. <kbd>Markdown 文档编写（本地）</kbd>：[VSCode](https://code.visualstudio.com/) | [Typora](https://typoraio.cn/)
10. <kbd>Markdown 文档编写（云端）</kbd>：[语雀](https://www.yuque.com/about)（邀请码 SGXMFL :smile:）
11. <kbd>回到桌面（不影响副屏）</kbd>：[ShowDesktopOneMonitor](https://github.com/ruzrobert/ShowDesktopOneMonitor)
12. <kbd>文本格式化工具</kbd>： [Reformat-Text-for-Clips](https://github.com/Le0v1n/Reformat-Text-for-Clips)
13. <kbd>将图片转换为图标</kbd>：[image2ico](https://github.com/Le0v1n/image2ico)
14. <kbd>模型权重查看</kbd>：[Netron(客户端)](https://github.com/lutzroeder/netron) | [Netron(Web版)](https://netron.app/)
15. <kbd>图形绘制</kbd>：[draw.io(客户端)](https://github.com/jgraph/drawio/releases) | [draw.io(Web)](https://app.diagrams.net/)
16. <kbd>剪切板历史</kbd>：[Ditto](https://ditto-cp.sourceforge.io/)
17. <kbd>截屏</kbd>：[Snipaste](https://www.snipaste.com/)
18. <kbd>Office 公式编写</kbd>：MathType
19. <kbd>邮箱</kbd>：[网易邮箱大师](https://dashi.163.com/download.html)
20. <kbd>本地搜索文件</kbd>：[Everthing](https://www.voidtools.com/zh-cn/)
21. <kbd>音乐播放下载器</kbd>：[lx-music-desktop](https://github.com/lyswhut/lx-music-desktop)
22. <kbd>视频播放器</kbd>：:star:[VLC](https://www.videolan.org/) | [PotPlayer](http://www.potplayercn.com/download)
23. <kbd>桌面助手</kbd>：[360桌面助手](http://www.360.cn/desktop/)
24. <kbd>云盘</kbd>：[阿里云盘](https://www.aliyundrive.com/)
25. <kbd>远程连接</kbd>：[To Desk](https://www.todesk.com/)
26. <kbd>待办事项</kbd>：[滴答清单(Web)](https://www.dida365.com/webapp/#q/all/tasks) | [滴答清单(客户端)](https://www.dida365.com/about/download)
27. <kbd>铃声制作</kbd>：[酷狗铃声制作专家(需下载酷狗音乐)](https://download.kugou.com/)
28. <kbd>视频剪辑</kbd>：[剪映专业版](https://www.capcut.cn/)
29. <kbd>手机与电脑传输文件</kbd>：[KDE Connect](https://kdeconnect.kde.org/) | [LANDrop](https://landrop.app/#downloads)
30. <kbd>平板充当电脑副屏</kbd>：[SpaceDesk](https://www.spacedesk.net/zh/)
31. <kbd>按空格快速预览文件</kbd>：[QuickLook](https://github.com/QL-Win/QuickLook)
32. <kbd>右键管理</kbd>：[ContextMenuManager](https://github.com/BluePointLilac/ContextMenuManager)
33. <kbd>Windows 10 音量模块修改</kbd>（<font color='red'>有 Bug：多媒体暂停和继续播放无效</font>）：[ModernFlyouts](https://github.com/ModernFlyouts-Community/ModernFlyouts)
34. <kbd>Windows 10 任务栏居中</kbd>：[StartIsBack++](https://www.ghxi.com/startisback.html)
35. <kbd>哔哩哔哩视频下载</kbd>：[BBDown](https://github.com/nilaoda/BBDown)


# 2. VSCode 配置
## 2.1 VSCode 插件

1. **autoDocstring**：在写 Python 代码的时候，输入 `"""` 会有提示，可以格式化的对函数、class 进行注释。
2. **CJK Word Handler**：在 VSCode 中，`Ctrl + ← / →` 时应该自动分词跳跃光标，但是 VSCode 默认只对英文有效，中文体验不好，这个插件可以对中文进行分词，从而提升编写代码的效率。
3. **CodeSnap**：用于分享代码（美化）。
4. **filesize**：可以帮我知道文件的大小。
5. **Markdown All in One**：用于写 Markdown。
6. **Markdown Preview Enhanced**：Markdown 文档的预览。
7. **Material Icon Theme**：好看的图标主题。
8. **Paste Image**：用于在 Markdown 中粘贴图片（需要配置）。
9. **Remote - SSH**：用于连接服务器。
10. **TabOut**：可以让 VSCode 实现像 PyCharm 那样，按下 `Tab` 键后自动跳过括号（`()``{}``[]` 均支持）。
11. **vscode-pdf**：让 VSCode 可以直接查看 `. pdf` 文件。

## 2.2 Paste Image 配置

在 `settings.json` 文件中写入以下语句：

```json
"pasteImage.path": "${currentFileDir}/imgs_markdown",
"pasteImage.basePath": "${currentFileDir}",
"pasteImage.forceUnixStyleSeparator": true,
"pasteImage.prefix": "/",
"pasteImage.insertPattern": "<div align=center>\n    <img src=./imgs_markdown/${imageFileName}\n    width=100%>\n</div>",
```

图片输出格式为：

```
<div align=center>
    <img src=./imgs_markdown/2023-10-18-11-04-05.png
    width=100%>
</div>
```

自动帮你进行图片的居中和缩放，如果不需要居中和缩放，可以修改为下面的：

```json
"pasteImage.path": "${currentFileDir}/imgs_markdown",
"pasteImage.basePath": "${currentFileDir}",
"pasteImage.forceUnixStyleSeparator": true,
"pasteImage.prefix": "/",
"pasteImage.insertPattern": "${imageSyntaxPrefix}./imgs_markdown/${imageFileName}${imageSyntaxSuffix}",
```

`imgs` 不喜欢可以替换。

这样再使用该插件往 Markdown 文件中插入图片就非常方便且易管理了。

## 2.3 VSCode 像 PyCharm 那样格式化 Python 代码
VSCode 中的 Python 格式化默认使用 autopep8 格式化代码，默认长度很短就会自动换行，用惯了 PyCharm 的朋友很可能不习惯，所以我们需要对其进行调整，使其与 PyCharm 对齐。

在 `setting.json` 添加设置：

```json
    "python.formatting.autopep8Args": [
        "--max-line-length=150"
    ],
```

若 autopep8 格式化无效，不起作用，可以添加 `--experimental` 参数：

```json
    "python.formatting.autopep8Args": ["--max-line-length", "150", "--experimental"],
```

> 建议在修改之后重启一下 VSCode

## 2.4 解决 VSCode 文件夹折叠问题

打开 VSCode 设置，搜索 `Explorer:Compact Folders`，将勾选取消。

# 3. 其他软件
## 3.1 draw.io 字体设置

```
Consolas, Monaco, "Courier New", monospace, Consolas, "Courier New", monospace
```

<div align=center>
<img src=https://img-blog.csdnimg.cn/3e2ae95b8ec14a9986aa74eb8db78293.png
width=40%>
</div>

