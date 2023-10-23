# 1. Word

## 1.1 标题设置为多级列表

选住**一级标题**，之后进行“定义新的多级列表”

<div align=center>
<div align=half>
    <img src=./imgs_markdown/2023-10-20-16-10-56.png width=30%>
    <img src=./imgs_markdown/2023-10-20-16-12-15.png width=50%>
</div>
</div>

## 1.2 图片和表的题注自动排序

正常插入题注后就可以了。如果一级标题是 “汉字序号”，那么需要对题注进行修改：

从原来的 `图 { STYLEREF 1 \s }-{ SEQ 图 \* ARABIC \s 1 }` 修改为 `图 { Quote “二零二五年一月{ STYLEREF 1 \s }日” \@”d” }-{ SEQ 图 \* ARABIC \s 1 }`

<kbd>注意</kbd>：之后别插入题注了，直接复制粘贴。

> 1. 目前还有很多 Bug :cry:
> 2. 也可以参考该视频的方法: [优雅搞定论文图片&表格](https://www.bilibili.com/video/BV1iZ4y1P7wa)

## 1.3 论文模板

我自己制作了一个模板，在同文件夹下。

## 1.4 自动排序的图表无法交叉引用

如果当我们在交叉引用的时候，发现没有自动排序的图表，如下图所示：

<div align=center>
    <img src=./imgs_markdown/2023-10-23-11-03-45.png
    width=50%>
</div>

那么我们随便找一找图片或者表格：<kbd>右键</kbd> → <kbd>插入题注(N)</kbd> → <kbd>新建标签(N)</kbd> → 输入`图`（或者`表`）→ <kbd>确定</kbd> → 不要点 <kbd>确定</kbd>，直接点 <kbd>×</kbd>。


我们用动图进行演示：

<div align=center>
    <img src=imgs_markdown\在题注中添加标签.gif
    width=100%>
</div>

之后我们再次使用交叉沿用，就可以看到刚刚我们新建的 `图`（或者`表`） 了。

<div align=center>
    <img src=./imgs_markdown/2023-10-23-11-34-06.png
    width=50%>
</div>

<kbd>注意</kbd>：
1. 我们的目的是新建一个标签，而不是插入题注，所以题注可以不用真的插进去，标签新建好了就行
2. 无论是插入图还是表的交叉引用，在正文中引用时，建议“引用内容”修改为“仅标签和编号”，这样得到的是 `图 2.1`，

# 2. Excel

# 3. PowerPoint (PPT)