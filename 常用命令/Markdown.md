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

### 3. 颜色表

<details>

<summary>展开/折叠(有点长)</summary>

> 在 Github 中可能无法正常显示字体颜色。

<div align="center">

| 颜色名 | 十六进制颜色值 | 颜色 |
|----------------------|--------------|-------------------|
| <font color="#F0F8FF">AliceBlue</font> | #F0F8FF | rgb(240, 248, 255) |
| <font color="#FAEBD7">AntiqueWhite</font> | #FAEBD7 | rgb(250, 235, 215) |
| <font color="#00FFFF">Aqua</font> | #00FFFF | rgb(0, 255, 255) |
| <font color="#7FFFD4">Aquamarine</font> | #7FFFD4 | rgb(127, 255, 212) |
| <font color="#F0FFFF">Azure</font> | #F0FFFF | rgb(240, 255, 255) |
| <font color="#F5F5DC">Beige</font> | #F5F5DC | rgb(245, 245, 220) |
| <font color="#FFE4C4">Bisque</font> | #FFE4C4 | rgb(255, 228, 196) |
| <font color="#000000">Black</font> | #000000 | rgb(0, 0, 0) |
| <font color="#FFEBCD">BlanchedAlmond</font> | #FFEBCD | rgb(255, 235, 205) |
| <font color="#0000FF">Blue</font> | #0000FF | rgb(0, 0, 255) |
| <font color="#8A2BE2">BlueViolet</font> | #8A2BE2 | rgb(138, 43, 226) |
| <font color="#A52A2A">Brown</font> | #A52A2A | rgb(165, 42, 42) |
| <font color="#DEB887">BurlyWood</font> | #DEB887 | rgb(222, 184, 135) |
| <font color="#5F9EA0">CadetBlue</font> | #5F9EA0 | rgb(95, 158, 160) |
| <font color="#7FFF00">Chartreuse</font> | #7FFF00 | rgb(127, 255, 0) |
| <font color="#D2691E">Chocolate</font> | #D2691E | rgb(210, 105, 30) |
| <font color="#FF7F50">Coral</font> | #FF7F50 | rgb(255, 127, 80) |
| <font color="#6495ED">CornflowerBlue</font> | #6495ED | rgb(100, 149, 237) |
| <font color="#FFF8DC">Cornsilk</font> | #FFF8DC | rgb(255, 248, 220) |
| <font color="#DC143C">Crimson</font> | #DC143C | rgb(220, 20, 60) |
| <font color="#00FFFF">Cyan</font> | #00FFFF | rgb(0, 255, 255) |
| <font color="#00008B">DarkBlue</font> | #00008B | rgb(0, 0, 139) |
| <font color="#008B8B">DarkCyan</font> | #008B8B | rgb(0, 139, 139) |
| <font color="#B8860B">DarkGoldenRod</font> | #B8860B | rgb(184, 134, 11) |
| <font color="#A9A9A9">DarkGray</font> | #A9A9A9 | rgb(169, 169, 169) |
| <font color="#006400">DarkGreen</font> | #006400 | rgb(0, 100, 0) |
| <font color="#BDB76B">DarkKhaki</font> | #BDB76B | rgb(189, 183, 107) |
| <font color="#8B008B">DarkMagenta</font> | #8B008B | rgb(139, 0, 139) |
| <font color="#556B2F">DarkOliveGreen</font> | #556B2F | rgb(85, 107, 47) |
| <font color="#FF8C00">Darkorange</font> | #FF8C00 | rgb(255, 140, 0) |
| <font color="#9932CC">DarkOrchid</font> | #9932CC | rgb(153, 50, 204) |
| <font color="#8B0000">DarkRed</font> | #8B0000 | rgb(139, 0, 0) |
| <font color="#E9967A">DarkSalmon</font> | #E9967A | rgb(233, 150, 122) |
| <font color="#8FBC8F">DarkSeaGreen</font> | #8FBC8F | rgb(143, 188, 143) |
| <font color="#483D8B">DarkSlateBlue</font> | #483D8B | rgb(72, 61, 139) |
| <font color="#2F4F4F">DarkSlateGray</font> | #2F4F4F | rgb(47, 79, 79) |
| <font color="#00CED1">DarkTurquoise</font> | #00CED1 | rgb(0, 206, 209) |
| <font color="#9400D3">DarkViolet</font> | #9400D3 | rgb(148, 0, 211) |
| <font color="#FF1493">DeepPink</font> | #FF1493 | rgb(255, 20, 147) |
| <font color="#00BFFF">DeepSkyBlue</font> | #00BFFF | rgb(0, 191, 255) |
| <font color="#696969">DimGray</font> | #696969 | rgb(105, 105, 105) |
| <font color="#1E90FF">DodgerBlue</font> | #1E90FF | rgb(30, 144, 255) |
| <font color="#D19275">Feldspar</font> | #D19275 | rgb(209, 146, 117) |
| <font color="#B22222">FireBrick</font> | #B22222 | rgb(178, 34, 34) |
| <font color="#FFFAF0">FloralWhite</font> | #FFFAF0 | rgb(255, 250, 240) |
| <font color="#228B22">ForestGreen</font> | #228B22 | rgb(34, 139, 34) |
| <font color="#FF00FF">Fuchsia</font> | #FF00FF | rgb(255, 0, 255) |
| <font color="#DCDCDC">Gainsboro</font> | #DCDCDC | rgb(220, 220, 220) |
| <font color="#F8F8FF">GhostWhite</font> | #F8F8FF | rgb(248, 248, 255) |
| <font color="#FFD700">Gold</font> | #FFD700 | rgb(255, 215, 0) |
| <font color="#DAA520">GoldenRod</font> | #DAA520 | rgb(218, 165, 32) |
| <font color="#808080">Gray</font> | #808080 | rgb(128, 128, 128) |
| <font color="#008000">Green</font> | #008000 | rgb(0, 128, 0) |
| <font color="#ADFF2F">GreenYellow</font> | #ADFF2F | rgb(173, 255, 47) |
| <font color="#F0FFF0">HoneyDew</font> | #F0FFF0 | rgb(240, 255, 240) |
| <font color="#FF69B4">HotPink</font> | #FF69B4 | rgb(255, 105, 180) |
| <font color="#CD5C5C">IndianRed</font> | #CD5C5C | rgb(205, 92, 92) |
| <font color="#4B0082">Indigo</font> | #4B0082 | rgb(75, 0, 130) |
| <font color="#FFFFF0">Ivory</font> | #FFFFF0 | rgb(255, 255, 240) |
| <font color="#F0E68C">Khaki</font> | #F0E68C | rgb(240, 230, 140) |
| <font color="#E6E6FA">Lavender</font> | #E6E6FA | rgb(230, 230, 250) |
| <font color="#FFF0F5">LavenderBlush</font> | #FFF0F5 | rgb(255, 240, 245) |
| <font color="#7CFC00">LawnGreen</font> | #7CFC00 | rgb(124, 252, 0) |
| <font color="#FFFACD">LemonChiffon</font> | #FFFACD | rgb(255, 250, 205) |
| <font color="#ADD8E6">LightBlue</font> | #ADD8E6 | rgb(173, 216, 230) |
| <font color="#F08080">LightCoral</font> | #F08080 | rgb(240, 128, 128) |
| <font color="#E0FFFF">LightCyan</font> | #E0FFFF | rgb(224, 255, 255) |
| <font color="#FAFAD2">LightGoldenRodYellow</font> | #FAFAD2 | rgb(250, 250, 210) |
| <font color="#D3D3D3">LightGrey</font> | #D3D3D3 | rgb(211, 211, 211) |
| <font color="#90EE90">LightGreen</font> | #90EE90 | rgb(144, 238, 144) |
| <font color="#FFB6C1">LightPink</font> | #FFB6C1 | rgb(255, 182, 193) |
| <font color="#FFA07A">LightSalmon</font> | #FFA07A | rgb(255, 160, 122) |
| <font color="#20B2AA">LightSeaGreen</font> | #20B2AA | rgb(32, 178, 170) |
| <font color="#87CEFA">LightSkyBlue</font> | #87CEFA | rgb(135, 206, 250) |
| <font color="#8470FF">LightSlateBlue</font> | #8470FF | rgb(132, 112, 255) |
| <font color="#778899">LightSlateGray</font> | #778899 | rgb(119, 136, 153) |
| <font color="#B0C4DE">LightSteelBlue</font> | #B0C4DE | rgb(176, 196, 222) |
| <font color="#FFFFE0">LightYellow</font> | #FFFFE0 | rgb(255, 255, 224) |
| <font color="#00FF00">Lime</font> | #00FF00 | rgb(0, 255, 0) |
| <font color="#32CD32">LimeGreen</font> | #32CD32 | rgb(50, 205, 50) |
| <font color="#FAF0E6">Linen</font> | #FAF0E6 | rgb(250, 240, 230) |
| <font color="#FF00FF">Magenta</font> | #FF00FF | rgb(255, 0, 255) |
| <font color="#800000">Maroon</font> | #800000 | rgb(128, 0, 0) |
| <font color="#66CDAA">MediumAquaMarine</font> | #66CDAA | rgb(102, 205, 170) |
| <font color="#0000CD">MediumBlue</font> | #0000CD | rgb(0, 0, 205) |
| <font color="#BA55D3">MediumOrchid</font> | #BA55D3 | rgb(186, 85, 211) |
| <font color="#9370D8">MediumPurple</font> | #9370D8 | rgb(147, 112, 216) |
| <font color="#3CB371">MediumSeaGreen</font> | #3CB371 | rgb(60, 179, 113) |
| <font color="#7B68EE">MediumSlateBlue</font> | #7B68EE | rgb(123, 104, 238) |
| <font color="#00FA9A">MediumSpringGreen</font> | #00FA9A | rgb(0, 250, 154) |
| <font color="#48D1CC">MediumTurquoise</font> | #48D1CC | rgb(72, 209, 204) |
| <font color="#C71585">MediumVioletRed</font> | #C71585 | rgb(199, 21, 133) |
| <font color="#191970">MidnightBlue</font> | #191970 | rgb(25, 25, 112) |
| <font color="#F5FFFA">MintCream</font> | #F5FFFA | rgb(245, 255, 250) |
| <font color="#FFE4E1">MistyRose</font> | #FFE4E1 | rgb(255, 228, 225) |
| <font color="#FFE4B5">Moccasin</font> | #FFE4B5 | rgb(255, 228, 181) |
| <font color="#FFDEAD">NavajoWhite</font> | #FFDEAD | rgb(255, 222, 173) |
| <font color="#000080">Navy</font> | #000080 | rgb(0, 0, 128) |
| <font color="#FDF5E6">OldLace</font> | #FDF5E6 | rgb(253, 245, 230) |
| <font color="#808000">Olive</font> | #808000 | rgb(128, 128, 0) |
| <font color="#6B8E23">OliveDrab</font> | #6B8E23 | rgb(107, 142, 35) |
| <font color="#FFA500">Orange</font> | #FFA500 | rgb(255, 165, 0) |
| <font color="#FF4500">OrangeRed</font> | #FF4500 | rgb(255, 69, 0) |
| <font color="#DA70D6">Orchid</font> | #DA70D6 | rgb(218, 112, 214) |
| <font color="#EEE8AA">PaleGoldenRod</font> | #EEE8AA | rgb(238, 232, 170) |
| <font color="#98FB98">PaleGreen</font> | #98FB98 | rgb(152, 251, 152) |
| <font color="#AFEEEE">PaleTurquoise</font> | #AFEEEE | rgb(175, 238, 238) |
| <font color="#D87093">PaleVioletRed</font> | #D87093 | rgb(216, 112, 147) |
| <font color="#FFEFD5">PapayaWhip</font> | #FFEFD5 | rgb(255, 239, 213) |
| <font color="#FFDAB9">PeachPuff</font> | #FFDAB9 | rgb(255, 218, 185) |
| <font color="#CD853F">Peru</font> | #CD853F | rgb(205, 133, 63) |
| <font color="#FFC0CB">Pink</font> | #FFC0CB | rgb(255, 192, 203) |
| <font color="#DDA0DD">Plum</font> | #DDA0DD | rgb(221, 160, 221) |
| <font color="#B0E0E6">PowderBlue</font> | #B0E0E6 | rgb(176, 224, 230) |
| <font color="#800080">Purple</font> | #800080 | rgb(128, 0, 128) |
| <font color="#FF0000">Red</font> | #FF0000 | rgb(255, 0, 0) |
| <font color="#BC8F8F">RosyBrown</font> | #BC8F8F | rgb(188, 143, 143) |
| <font color="#4169E1">RoyalBlue</font> | #4169E1 | rgb(65, 105, 225) |
| <font color="#8B4513">SaddleBrown</font> | #8B4513 | rgb(139, 69, 19) |
| <font color="#FA8072">Salmon</font> | #FA8072 | rgb(250, 128, 114) |
| <font color="#F4A460">SandyBrown</font> | #F4A460 | rgb(244, 164, 96) |
| <font color="#2E8B57">SeaGreen</font> | #2E8B57 | rgb(46, 139, 87) |
| <font color="#FFF5EE">SeaShell</font> | #FFF5EE | rgb(255, 245, 238) |
| <font color="#A0522D">Sienna</font> | #A0522D | rgb(160, 82, 45) |
| <font color="#C0C0C0">Silver</font> | #C0C0C0 | rgb(192, 192, 192) |
| <font color="#87CEEB">SkyBlue</font> | #87CEEB | rgb(135, 206, 235) |
| <font color="#6A5ACD">SlateBlue</font> | #6A5ACD | rgb(106, 90, 205) |
| <font color="#708090">SlateGray</font> | #708090 | rgb(112, 128, 144) |
| <font color="#FFFAFA">Snow</font> | #FFFAFA | rgb(255, 250, 250) |
| <font color="#00FF7F">SpringGreen</font> | #00FF7F | rgb(0, 255, 127) |
| <font color="#4682B4">SteelBlue</font> | #4682B4 | rgb(70, 130, 180) |
| <font color="#D2B48C">Tan</font> | #D2B48C | rgb(210, 180, 140) |
| <font color="#008080">Teal</font> | #008080 | rgb(0, 128, 128) |
| <font color="#D8BFD8">Thistle</font> | #D8BFD8 | rgb(216, 191, 216) |
| <font color="#FF6347">Tomato</font> | #FF6347 | rgb(255, 99, 71) |
| <font color="#40E0D0">Turquoise</font> | #40E0D0 | rgb(64, 224, 208) |
| <font color="#EE82EE">Violet</font> | #EE82EE | rgb(238, 130, 238) |
| <font color="#F5DEB3">Wheat</font> | #F5DEB3 | rgb(245, 222, 179) |
| <font color="#FFFFFF">White</font> | #FFFFFF | rgb(255, 255, 255) |
| <font color="#F5F5F5">WhiteSmoke</font> | #F5F5F5 | rgb(245, 245, 245) |
| <font color="#FFFF00">Yellow</font> | #FFFF00 | rgb(255, 255, 0) |
| <font color="#9ACD32">YellowGreen</font> | #9ACD32 | rgb(154, 205, 50) |

</div>

</details>