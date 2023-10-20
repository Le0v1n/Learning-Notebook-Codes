### 1. Markdown 基础语法

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
| &nbsp;你好 |`&nbsp;你好`| 缩进 1/2 英文(半个字符) |
| &ensp;你好 |`&ensp;你好`| 缩进 1/1 英文(一个字符) |
| &emsp;你好 |`&emsp;你好`| 缩进 2/1 中文(二个字符) |

</div>

[^脚注的名字]: 这是一个演示的脚注（脚注的内容）

### 2. Markdown 高级语法

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