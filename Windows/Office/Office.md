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

，链接为：[论文模板.docx](https://github.com/Le0v1n/Learning-Notebook-Codes/blob/main/Windows/Office/%E8%AE%BA%E6%96%87%E6%A8%A1%E6%9D%BF.docx)

<kbd>注意</kbd>：

1. 直接复制这个 `.docx` 文档中的图片和表格的题注是没有任何 Bug 的！
2. 如果无法交叉沿用，那么请参考下一节内容。

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
    width=80%>
</div>

之后我们再次使用交叉沿用，就可以看到刚刚我们新建的 `图`（或者`表`） 了。

<div align=center>
    <img src=./imgs_markdown/2023-10-23-11-34-06.png
    width=50%>
</div>

<kbd>注意</kbd>：
1. 我们的目的是新建一个标签，而不是插入题注，所以题注可以不用真的插进去，标签新建好了就行
2. 无论是插入图还是表的交叉引用，在正文中引用时，建议“引用内容”修改为“仅标签和编号”，这样得到的是 `图 2.1`，

## 1.5 Word 连续引用 ^[1][3]^ ^→^ ^[1~3]^

如下图所示，我们目前的格式如下：

<div align=center>
    <img src=./imgs_markdown/2023-10-24-09-48-29.png
    width=80%>
</div>

**动图演示**：

<div align=center>
    <img src=./imgs_markdown/连续引用.gif
    width=80%>
</div>

下面是详细步骤 :smile:。

### 1.5.1 Step 1

之后我们选住引用 → <kbd>右键</kbd> → <kbd>切换域代码</kbd>：

<div align=center>
    <img src=./imgs_markdown/2023-10-24-09-49-40.png
    width=50%>
</div>

得到如下所示的域代码：

<div align=center>
    <img src=./imgs_markdown/2023-10-24-09-50-13.png
    width=80%>
</div>

### 1.5.2 Step 2

在第一个`\h `的后面加上`\#"[0"`，在最后一个`\h `的后面加上`\#"0]"`

+ **修改前**：
  ```
  { REF_Ref149033284 \r \h  \* MERGEFORMAT }{ REF_Ref149033288 \r \h  \* MERGEFORMAT }
  ```

+ **修改后**：
  ```
  { REF_Ref149033284 \r \h \#"[0" \* MERGEFORMAT }{ REF_Ref149033288 \r \h \#"0]" \* MERGEFORMAT }
  ```

<div align=center>
    <img src=./imgs_markdown/2023-10-24-09-55-02.png
    width=80%>
</div>

### 1.5.3 Step 3

再次选中域代码 → 按 <kbd>F9</kbd>

<div align=center>
    <img src=./imgs_markdown/2023-10-24-09-56-13.png
    width=80%>
</div>

为了显示正常，我们可以在 `1` 和 `4` 中间加上 `-` 或 `~`，如下所示：

<div align=center>
    <img src=./imgs_markdown/2023-10-24-09-58-56.png
    width=50%>
</div>

---

<kbd>Note</kbd>:

1. 该方法适用于 ^[1][3]^ ^→^ ^[1~3]^ 外，还适用于 ^[1][2]^ ^→^ ^[1,2]^
2. 执行完 Step 2 之后，一定要“更新域”
3. 更新域也可以 <kbd>Ctrl + A</kbd> → <kbd>F9</kbd>，这样会更新整篇论文的“域代码”（包括自动编号的图表题注、MathType、自动生成的目录等）

# 2. Excel

# 3. PowerPoint (PPT)