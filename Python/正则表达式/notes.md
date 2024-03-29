# 1. 正则表达式的介绍

正则表达式（Regular Expression，简称：Regex）是一种文本模式的表示方法，它使用单个字符串来描述、匹配一系列符合某个句法规则的字符串。正则表达式常用于字符串的搜索、替换以及复杂的字符串模式匹配操作。

正则表达式由普通字符（例如字符 `a` 到 `z`）和特殊字符（称为元字符）组成，它在不同的编程语言和工具中有着广泛的应用，例如 Python、JavaScript、Perl、Java 等语言，以及文本编辑器、搜索工具等软件中。

正则表达式的能力非常强大，可以用于各种复杂的字符串处理任务，如数据验证、数据提取、数据替换等。然而，编写复杂的正则表达式可能比较困难，需要一定的学习和实践。

# 2. 小例子

一个文本文件里面存储了一些市场职位信息，格式如下所示：

```
Python3 高级开发工程师 上海互教教育科技有限公司上海-浦东新区2万/月02-18满员
测试开发工程师(C++/python) 上海墨鹍数码科技有限公司上海-浦东新区2.5万/每月02-18未满员
Python3 开发工程师 上海德拓信息技术股份有限公司上海-徐汇区1.3万/每月02-18剩余11人
测试开发工程师(Python) 赫里普(上海)信息科技有限公司上海-浦东新区1.1万/每月02-18剩余5人
Python高级开发工程师 上海行动教育科技股份有限公司上海-闵行区2.8万/月02-18剩余255人
python开发工程师 上海优似腾软件开发有限公司上海-浦东新区2.5万/每月02-18满员
```

现在，我们需要写一个程序，从这些文本里面抓取所有职位的薪资。

这是典型的字符串处理。分析这里面的规律，可以发现，薪资的数字后面都有关键字 `万/月` 或者 `万/每月`。根据我们学过的知识，我们不难写出下面的代码：

```python
# 打开指定文本，获取文本中的内容
with open('Python/正则表达式/code/exp1.txt', 'r') as f:
    lines = f.readlines()

# 遍历每一行
for line in lines:
    # 查找'万/月'在字符串中的索引
    pos2 = line.find('万/月')  # 如果没有找到则返回-1
    
    if pos2 == -1:  # 说明 '万/月' 没有找到
        pos2 = line.find('万/每月')  # 查找 '万/每月' 在字符串中的索引
        
        if pos2 == -1:  # 说明 '万/每月' 也没有找到
            continue
    
    # 找到了 '万/月' 或者 '万/每月'
    idx = pos2 - 1  # 数字的末尾索引
    
    # 往前找
    while line[idx].isdigit() or line[idx] == '.':  # 如果前面的是数字或者小数点
        idx -= 1  # 继续往前找
    
    # 现在我们可以确定数字的索引范围了
    pos1 = idx + 1
    
    print(line[pos1: pos2], '万/每月')
```

运行结果：

```
2 万/每月
2.5 万/每月
1.3 万/每月
1.1 万/每月
2.8 万/每月
2.5 万/每月
```

> 💡  关于 `'Python/正则表达式/code/exp1.txt'`：自己新建一个 `.txt` 文本，将内容放入文件即可。

为了从每行获取薪资对应的数字，我们写了不少行代码，这种`从字符串中搜索出某种特征的子串`有没有更简单的方法呢？解决方案就是我们今天要介绍的正则表达式（Regular Expression）。如果我们使用正则表达式，代码可以这样：

```python
import re  # re: regular expression


# 打开指定文本，获取文本中的内容
with open('Python/正则表达式/code/exp1.txt', 'r') as f:
    lines = f.readlines()
    text = ''.join(lines)  # 使用 join 方法将列表转换为字符串
    
p = re.compile('([\d.]+)万/每{0,1}月')

for one in p.findall(text):
    print(f"{one} 万/月")
```

```
2 万/每月
2.5 万/每月
1.3 万/每月
1.1 万/每月
2.8 万/每月
2.5 万/每月
```

`p = re.compile('([\d.]+)万/每{0,1}月')` 的组成部分的解释：

- `re.compile`: 这是 Python 中 `re` 模块的一个函数，用于编译一个正则表达式模式，返回一个模式（pattern）对象，该对象可以用于匹配文本。
- `([\d.]+)`: 这是一个捕获组（group），用于匹配一个或多个数字（`\d` 表示任何数字，包括 0-9）或小数点（`.`）。`+` 表示匹配前面的子表达式一次或多次。这个捕获组将匹配一个数字序列，可能包含小数点，例如 `2.5`。
- `万/每{0,1}月`: 这部分匹配文本“万/每月”或“万/月”。这里使用了量词 `{0,1}`，它表示前面的字符“每”可以出现 0 次或 1 次。这意味着既可以匹配“万/每月”，也可以匹配“万/月”。

# 3. 在线验证

怎么验证我们写的表达式是否能正确匹配到要搜索的字符串呢？可以访问 [regex101](https://regex101.com/)，从而进行在线验证。[Fig.1](#Fig.1) 即为示意图。

<a id=Fig.1></a>
<div align=center>
    <img src=./imgs_markdown/2024-03-28-15-34-24.png
    width=100%>
    <center>Fig.1 regex101 示例图</center>
</div></br>

# 4. Regular Expression 常见语法

## 4.1 普通字符

写在正则表达式里面的普通字符都是表示： **直接匹配它们**。比如在下面的文本中，如果我们要找所有的 `test`， 正则表达式就非常简单，直接输入 `test` 即可。如 [Fig.2](#Fig.2) 所示：

<a id=Fig.2></a>
<div align=center>
    <img src=./imgs_markdown/2024-03-28-15-46-09.png
    width=100%>
    <center>Fig.2 直接匹配示例图</center>
</div></br>

汉字也是一样，要寻找汉字，直接写在正则表达式里面就可以了。

## 4.2 特殊字符（Meta Characters）

但是有些特殊的字符，术语叫 `metacharacters`（元字符）。它们出现在正则表达式字符串中，不是表示直接匹配他们，而是表达一些特别的含义。

> 关于 Meta 的介绍可以参考：[什么是 metadata（元数据、meta、metadata、诠释资料、元资料）](https://blog.csdn.net/weixin_44878336/article/details/135771574)

这些特殊的元字符及其含义如 [Table.1](#Table.1) 所示。

<center>Table.1 特殊字符总览</center><a id=Table.1></a>

| 符号 | 描述 |
|:-:|------|
| `.` | 匹配除换行符 `\n` 之外的任意单个字符。 |
| `*` | 匹配前面的元素零次或多次。 |
| `+` | 匹配前面的元素一次或多次。 |
| `?` | 匹配前面的元素零次或一次。在量词后使用时，可以使量词变为非贪婪模式。 |
| `\` | 转义字符，用于匹配特殊字符，让它们被视为普通字符。 |
| `[]` | 用于创建字符类，匹配方括号内的任意一个字符。 |
| `^` | 匹配行的开始。在字符类内部使用时，表示取非，匹配不在方括号内的任意一个字符。 |
| `$` | 匹配行的结束。 |
| `{m,n}` | 匹配前面的元素至少 `m` 次，至多 `n` 次。 |
| `\|` | 逻辑或操作符，匹配符号前后的表达式中的任意一个。 |
| `()` | 分组操作符，将括号内的表达式作为一个整体进行处理，并可以捕获匹配的子字符串以供后续使用。 |

接下来我们分别介绍这些特殊字符。

### 4.2.1 特殊字符: `.`（通配符：匹配除换行符外的所有字符）

`.` 又名通配符，表示要匹配除了 `换行符 \n`之外的任意 `单个` 字符。比如，要从下面的文本中，选择出所有的颜色。

```
苹果是绿色的
橙子是橙色的
香蕉是黄色的
乌鸦是黑色的
```

也就是要找到所有以 `色` 结尾，并且包括前面的一个字符的 `词语`。就可以这样写正则表达式：

```regex
.色
```

其中 `.` 代表了任意的**一个**字符。

`.色` 合起来就表示要找任意一个字符后面是 `色` 这个字， 合起来两个字的字符串。验证一下，如 [Fig.3](#Fig.3) 所示：

<a id=Fig.3></a>
<div align=center>
    <img src=./imgs_markdown/2024-03-28-16-01-40.png
    width=45%>
    <center>Fig.3 .色示例图</center>
</div></br>

我们也在 Python 中以代码的形式演示一下：

```python
import re


content = """苹果是绿色的
橙子是橙色的
香蕉是黄色的
乌鸦是黑色的
"""
    
pattern = re.compile(pattern=".色")

for item in pattern.findall(string=content):
    print(item)
```

```
绿色
橙色
黄色
黑色
```

高亮行 `re.compile` 函数返回的是正则表达式对象，它对应的匹配规则在参数中进行定义。之后我们可以调用这个对象的 `findall` 方法来进行正则表达式搜索操作。

💡  如果我们只是一次性使用这个匹配规则，可以不用先 `re.compile` 产生正则表达式对象，可以直接使用 `re.findall` 函数，相当于我们直接调用了一个静态方法（函数）。

```python
import re


content = """苹果是绿色的
橙子是橙色的
香蕉是黄色的
乌鸦是黑色的
"""

for item in re.findall(pattern='.色', 
                       string=content):
    print(item)
```

```
绿色
橙色
黄色
黑色
```

<kbd><b>Question</b></kbd>：特殊字符 `.` 可以匹配除了 `\n` 外的特殊字符吗？

<kbd><b>Answer</b></kbd>：特殊字符 `.` 在正则表达式中被称为“通配符”，它能够匹配除换行符 `\n` 以外的任意单个字符，包括特殊字符。例如，如果我们的正则表达式是 `a.c`，它将匹配以下字符串：

- `abc`
- `a*c`
- `a@c`
- `a c` (其中 ` ` 是一个空格)

在上述例子中，`.` 能够匹配任意单个字符，包括特殊字符如 `*`、`@` 和空格。需要注意的是，`.` 对换行符 `\n` 不起作用，如果我们想要匹配包括换行符在内的任意字符，需要使用特殊的模式或者标志，如在 Python 的 `re` 模块中使用 `re.DOTALL` 标志。

> 💡  `.` 也是可以匹配 `.` 的 🤣


### 4.2.2 特殊字符：`*`（重复匹配任意次）

`*` 表示匹配前面的子表达式任意次，**包括 0 次**。比如，我们要从下面的文本中，选择每行逗号后面的字符串内容，包括逗号本身。注意，这里的逗号是中文的逗号。

```
苹果，是绿色的
橙子，是橙色的
香蕉，是黄色的
乌鸦，是黑色的
猴子，
```

那我们就可以这样写正则表达式：

```regex
，.*
```

`*` 紧跟在 `.` 后面，意思是“匹配前面的子表达式任意次”，其中的子表达式是 `.`，表示除了换行符外的任意字符，所以 `.*` 表示前面的任意字符。而 `，` 就是普通字符匹配，没有什么好说的。因此整个表达式的意思就是在逗号后面的所有字符，包括逗号。我们验证一下，结果如 [Fig.4](#Fig.4)：

<a id=Fig.4></a>
<div align=center>
    <img src=./imgs_markdown/2024-03-28-16-24-52.png
    width=30%>
    <center>Fig.4 ，.* 示例图</center>
</div></br>

特别是最后一行，`猴子，`后面没有其它字符了，但是 `*` 表示可以匹配 0 次， 所以表达式也是成立的。我们也用 Python 代码看一下：

```python
import re


content = """苹果，是绿色的
橙子，是橙色的
香蕉，是黄色的
乌鸦，是黑色的
猴子，
"""

for item in re.findall(pattern='，.*', 
                       string=content):
    print(item)
```

```
，是绿色的
，是橙色的
，是黄色的
，是黑色的
，
```

💡  **注意**， `.*` 在正则表达式中非常常见，表示匹配任意字符任意次数。

当然这个 `*` 前面不是非得是 `.`，也可以是其它字符，如 [Fig.5](#Fig.5) 所示。

<a id=Fig.5></a>
<div align=center>
    <img src=./imgs_markdown/2024-03-28-16-37-18.png
    width=80%>
    <center>Fig.5 其他* 的示例</center>
</div></br>

### 4.2.3 特殊字符：`+`（重复匹配多次，不包括 0 次）

`+` 表示匹配前面的子表达式**一次或多次**，<font color='red'><b>不包括 0 次</b></font>。

还是上面的例子，我们要从文本中，选择每行逗号后面的字符串内容，包括逗号本身。但是添加一个条件，如果逗号后面没有内容，就不要选择了。

```
苹果，是绿色的
橙子，是橙色的
香蕉，是黄色的
乌鸦，是黑色的
猴子，
```

我们可以这样写正则表达式：

```regex
，.+
```

验证结果如 [Fig.6](#Fig.6) 所示。

<a id=Fig.6></a>
<div align=center>
    <img src=./imgs_markdown/2024-03-28-16-51-16.png
    width=80%>
    <center>.+ 的使用示例</center>
</div></br>

最后一行，`猴子，` 后面没有其它字符了，`+` 表示至少匹配 1 次， 所以最后一行没有子串选中。

### 4.2.4 特殊字符：`?`（匹配 0 ~ 1 次）

`?` 表示匹配前面的子表达式 0 次或 1 次。

还是上面的例子，我们要从文本中，选择每行逗号后面的1个字符，也包括逗号本身。

```
苹果，绿色的
橙子，橙色的
香蕉，黄色的
乌鸦，黑色的
猴子，
```

那正则表达式可以这样写：

```regex
，.?
```

验证结果如 [Fig.7](#Fig.7) 所示。

<a id=Fig.7></a>
<div align=center>
    <img src=./imgs_markdown/2024-03-28-16-54-15.png
    width=80%>
    <center>.? 的使用示例</center>
</div></br>

最后一行，`猴子，` 后面没有其它字符了，但是 `?`` 表示匹配 1 次或 0 次， 所以最后一行也选中了一个逗号字符。

### 4.2.5 特殊字符：`{}`（匹配指定次数）

示例文本如下所示：

```
红彤彤，绿油油，黑乎乎，绿油
红彤彤，绿油油，黑乎乎，绿油油
红彤彤，绿油油，黑乎乎，绿油油油
红彤彤，绿油油，黑乎乎，绿油油油油
红彤彤，绿油油，黑乎乎，绿油油油油油
```

- 表达式 `油{3}` 就表示匹配**连续的** `油` 字 3 次。
- 表达式 `油{3,4}` 就表示匹配**连续的** `油` 字 至少 3 次，至多 4 次

示例如 [Fig.8](#Fig.8) 所示。

<a id=Fig.8></a>
<div align=center>
    <img src=./imgs_markdown/2024-03-28-17-00-07.png
    width=80%>
    <center>字符{int} 和 字符{int,int} 的使用示例</center>
</div></br>

### 4.2.6 特殊字符：`\`（转义字符）

反斜杠 `\` 用作转义字符，它有两种作用：

1. 用于转义紧跟其后的特殊字符，使其失去特殊含义，被当作普通字符对待。
2. 用于创建一些特定的字符类，如换行符 `\n`、制表符 `\t` 等。

例如，如果我们想匹配一个实际的点 `.`，而不是作为通配符的点，我们需要在点前面加上反斜杠 `\.`。同样，如果我们想匹配一个实际的反斜杠 `\`，我们需要使用两个反斜杠 `\\`，因为在字符串中反斜杠本身也是一个转义字符。

---

我们的示例文本如下：

```
example.com
example.net
example.org
```

现在我们想要得到 `.` 后面的后缀以确定网页的类型（包括 `.` 本身），那么我们的 Regex 可以这样写：

```regex
\..*
```

我们验证一下，验证结果见 [Fig.9](#Fig.9)。

<a id=Fig.9></a>
<div align=center>
    <img src=./imgs_markdown/2024-03-28-17-08-04.png
    width=80%>
    <center>Fig.9 转义字符 \ 的使用示例</center>
</div></br>

在这里，`\.` 表示 `.` 本身，它是一个普通字符而非特殊字符。`.*` 表示除了换行符外的所有字符。

我们也用 Python 进行一下验证：

```python
import re


text = "这是一个例子：example.com"
pattern = r"example\.com"
match = re.search(pattern, text)

if match:
    print(f"匹配结果：{match.group()}")
else:
    print("没有匹配结果")

# 匹配包含反斜杠的文本
text_with_backslash = "路径：C:\\Program Files\\Example"
pattern_with_backslash = r"C:\\Program Files\\Example"
match_with_backslash = re.search(pattern_with_backslash, text_with_backslash)

if match_with_backslash:
    print(f"匹配结果：{match_with_backslash.group()}")
else:
    print("没有匹配结果")
```

```
匹配结果：example.com
匹配结果：C:\Program Files\Example
```

---

<kbd><b>Question</b></kbd>：为什么要加 `r` ？

<kbd><b>Answer</b></kbd>：在 Python 中，字符串前加上 `r` 或 `R` 表示这是一个原始字符串（raw string）。在原始字符串中，反斜杠 `\` 不会被当作转义字符处理，而是保持其字面意义。这意味着在原始字符串中，反斜杠后面的字符不会被特殊解释。💡  如果我们不使用原始字符串，我们需要写四个反斜杠 `\\\\` 来表示一个反斜杠  🤣。

### 4.2.7 特殊字符：`[]`（匹配字符集中任意字符）

方括号 `[]` 用于创建一个字符集，匹配方括号内列出的任意一个字符。字符集可以包含普通字符和特殊字符，<font color='red'><b>但特殊字符在字符集中将失去其特殊含义，被视为普通字符</b></font>。

例如，字符集 `[abc]` 将匹配字母 `a`、`b` 或 `c` 中的任意一个。字符集也可以包含字符范围，如 `[a-z]` 将匹配从小写 `a` 到小写 `z` 的任意字母。

如果字符集的第一个字符是脱字符 `^`，则表示取非，匹配任何不在方括号内的字符。例如，`[^abc]` 将匹配除了 `a`、`b` 和 `c` 之外的任意字符。

---

我们的示例文本如下：

```
abc def ghi jkl mno a b c d aa  a a a a a a sdsad sajkjclkx jsadkl dskljnsdlijewqlkjsadj lasdjlkjdwijsalkj lksajd lkasjwd
```

- 现在我们想要得到字符 ace，那么我们的 Regex 可以这样写：
    ```regex
    [ace]
    ```
- 如果我们不想要字符 ace，那么我们的 Regex 可以这样写：
    ```regex
    [^ace]
    ```
- 如果我们想要字符 a 到 e 范围内的所有字符，那么我们的 Regex 可以这样写：
    ```regx
    [a-e]
    ```

验证如 [Fig.10](#Fig.10) 所示。

<a id=Fig.10></a>
<div align=center>
    <img src=./imgs_markdown/2024-03-28-17-28-50.png
    width=80%>
    <center>Fig.10 字符集的使用示例</center>
</div></br>

可以看到，当我们不想要字符 `ace` 时（`[ace]`），空格也被包围了，很合理。当 `[a-e]` 时，空格并没有被选中，也非常合理。


`[]` 的 Python 示例：
```python
import re


# 匹配字符集内的任意字符
text = "abc def ghi jkl mno a b c d aa  a a a a a a sdsad sajkjclkx jsadkl dskljnsdlijewqlkjsadj lasdjlkjdwijsalkj lksajd lkasjwd"
pattern = r"[ace]"
matches = re.findall(pattern, text)
print(f"匹配结果：{matches}")

# 匹配不在字符集内的任意字符
text = "abc def ghi jkl mno a b c d aa  a a a a a a sdsad sajkjclkx jsadkl dskljnsdlijewqlkjsadj lasdjlkjdwijsalkj lksajd lkasjwd"
pattern = r"[^ace]"
matches = re.findall(pattern, text)
print(f"匹配结果：{matches}")

# 匹配字符范围
text = "abc def ghi jkl mno a b c d aa  a a a a a a sdsad sajkjclkx jsadkl dskljnsdlijewqlkjsadj lasdjlkjdwijsalkj lksajd lkasjwd"
pattern = r"[a-e]"
matches = re.findall(pattern, text)
print(f"匹配结果：{matches}")
```

```
匹配结果：['a', 'c', 'e', 'a', 'c', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'c', 'a', 'e', 'a', 'a', 'a', 'a', 'a']
匹配结果：['b', ' ', 'd', 'f', ' ', 'g', 'h', 'i', ' ', 'j', 'k', 'l', ' ', 'm', 'n', 'o', ' ', ' ', 'b', ' ', ' ', 'd', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 's', 'd', 's', 'd', ' ', 's', 'j', 'k', 'j', 'l', 'k', 'x', ' ', 'j', 's', 'd', 'k', 'l', ' ', 'd', 's', 'k', 'l', 'j', 'n', 's', 'd', 'l', 'i', 'j', 'w', 'q', 'l', 'k', 'j', 's', 'd', 'j', ' ', 'l', 's', 'd', 'j', 'l', 'k', 'j', 'd', 'w', 'i', 'j', 's', 'l', 'k', 'j', ' ', 'l', 'k', 's', 'j', 'd', ' ', 'l', 'k', 's', 'j', 'w', 'd']
匹配结果：['a', 'b', 'c', 'd', 'e', 'a', 'b', 'c', 'd', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'd', 'a', 'd', 'a', 'c', 'a', 'd', 'd', 'd', 'e', 'a', 'd', 'a', 'd', 'd', 'a', 'a', 'd', 'a', 'd']
```

在这个例子中，`re.findall` 函数用于找到所有匹配正则表达式的子串。第一个模式 `[ace]` 匹配文本中所有的 `a`、`c` 和 `e`。第二个模式 `[^ace]` 匹配除了 `a`、`c` 和 `e` 之外的字符。第三个模式 `[a-e]` 匹配从 `a` 到 `e` 的所有字母。

### 4.2.8 特殊字符：`^`（锚定匹配字符串的开头和脱字符）

脱字符 `^` 有两种用途：

1. 〔**锚定匹配字符串的开头**〕当 `^` 出现在正则表达式的开头时，它表示匹配行的开始。也就是说，它指定接下来的模式必须出现在被搜索字符串的开头。
2. 〔**脱字符**〕当 `^` 出现在字符集的方括号 `[]` 内时，它表示取非，用于排除字符集内的字符。在这种情况下，它匹配任何不在方括号内列出的字符（我们刚刚接触过）。

💡  这里对第一种作用再次阐述：当 `^` 出现在模式的开头时，它表示锚定（anchoring）作用，用于指定匹配必须发生在被搜索字符串的开始位置。也就是说，只有当被搜索的字符串以这个模式开始时，匹配才会成功。

也就是说 `^` 表示匹配文本的<font color='red'><b>开头位置</b></font>。正则表达式可以设定单行模式和多行模式。

- 如果是单行模式，`^` 表示匹配整个文本的**开头位置**。
- 如果是多行模式，`^` 表示匹配文本每行的**开头位置**。

比如，下面的文本中，每行最前面的数字表示水果的编号，最后的数字表示价格。

```
001-苹果价格-60
002-橙子价格-70
003-香蕉价格-80
```

如果我们要提取所有的水果编号，用这样的正则表达式：

```regex
^\d+
```

其中：

- `\d` 可以先看一下 [4.4 匹配某种字符类型](#44-匹配某种字符类型)，简单来说：`\d` 匹配 0-9 之间任意一个数字字符，等价于表达式 `[0-9]`。
- `+` 表示至少匹配一次

那此时我们有点疑问，如果是 `\d+`，那意味着会至少匹配一次及以上的数字，那么它不仅会匹配编号，也会匹配价格，我们用例子看一下：

```python
import re


text = """
001-苹果价格-60
002-橙子价格-70
003-香蕉价格-80
"""

pattern = r"[\d+]"
matches = re.findall(pattern, text)

print(f"{matches = }")
```

```
matches = ['0', '0', '1', '6', '0', '0', '0', '2', '7', '0', '0', '0', '3', '8', '0']
```

可以看出来，我们的想法是正确的。那么我们怎么才能只匹配一次呢？设定为不贪婪的模式？[4.3-贪婪模式和非贪婪模式](#43-贪婪模式和非贪婪模式)？我们实际看一下：

```python
import re


text = """
001-苹果价格-60
002-橙子价格-70
003-香蕉价格-80
"""

pattern = r"[\d+?]"
matches = re.findall(pattern, text)

print(f"{matches = }")
```

```
matches = ['0', '0', '1', '6', '0', '0', '0', '2', '7', '0', '0', '0', '3', '8', '0']
```

错了！为什么？

这是因为我们使用的是 `[]` 进行的匹配（`[]` 表示匹配字符集中任意字符），在 `[]` 中，所有的特殊字符会被当做普通字符，所以 `?` 没有开启不贪婪模式。那我们怎么办？去掉 `[]` ？我们可以试一下：

```python
text = """
001-苹果价格-60
002-橙子价格-70
003-香蕉价格-80
"""

pattern = r"\d+"
matches = re.findall(pattern, text)

print(f"{matches = }")
```

```
matches = ['001', '60', '002', '70', '003', '80']
```

我们发现这样不仅编号被匹配了，价钱也被匹配了。那我们让其不再贪婪试试？

```python
text = """
001-苹果价格-60
002-橙子价格-70
003-香蕉价格-80
"""

pattern = r"\d+?"
matches = re.findall(pattern, text)

print(f"{matches = }")
```

```
matches = ['0', '0', '1', '6', '0', '0', '0', '2', '7', '0', '0', '0', '3', '8', '0']
```

确实是不贪婪了，但还是会匹配所有的数字 🤣。

到这里我们就要使用本节的主角 `^` 了。它的作用是让匹配范围只在每行的开头，我们试一下：

```python
import re


text = """
001-苹果价格-60
002-橙子价格-70
003-香蕉价格-80
"""

pattern = r"^\d+"
matches = re.findall(pattern, text)

print(f"{matches = }")
```

```
matches = []
```

这是什么情况，为什么没有匹配到任何内容？出现这种原因是因为我的 `text` 有问题，如上的方式并不是从第一行开始的，而是从第二行开始的，正确的写法应该是：

```python
text = """001-苹果价格-60
002-橙子价格-70
003-香蕉价格-80"""
```

这样才是正确的 3 行，否则为 5 行！我们用代码看一下：

```python
text = """
001-苹果价格-60
002-橙子价格-70
003-香蕉价格-80
"""

print(f"{text}")
```

结果如 [Fig.11](#Fig.11) 所示：

<a id=Fig.11></a>
<div align=center>
    <img src=./imgs_markdown/2024-03-29-09-58-13.png
    width=100%>
    <center>Fig.11 """""" 的错误使用</center>
</div></br>

我们再看一下 3 行的写法和效果：

```python
text = """001-苹果价格-60
002-橙子价格-70
003-香蕉价格-80"""

print(text)
```

结果见 [Fig.12](#Fig.12)。

<a id=Fig.12></a>
<div align=center>
    <img src=./imgs_markdown/2024-03-29-10-00-13.png
    width=100%>
    <center>Fig.12 """""" 的正确使用</center>
</div></br>

那我们接下来再看一下在正确使用 `""""""` 的情况下 Regex 的效果：

```python
import re


text = """001-苹果价格-60
002-橙子价格-70
003-香蕉价格-80"""

pattern = r"^\d+"
matches = re.findall(pattern, text)

print(f"{matches = }")
```

```
matches = ['001']
```

为什么只匹配了一个？这是因为我们使用的是单行模式，如果需要使用多行模式，则：

```python
import re


text = """001-苹果价格-60
002-橙子价格-70
003-香蕉价格-80"""

pattern = r"^\d+"
matches = re.findall(pattern, text, re.M)  # 传入参数 re.M 则开启多行模式

print(f"{matches = }")
```

```
matches = ['001', '002', '003']
```

再写一个简单的例子：

```python
import re


# 示例字符串
text1 = "hello world"
text2 = "say hello world"

# 正则表达式，用于匹配以 "hello" 开始的字符串
pattern = r"^hello"

# 使用 re.match 检查匹配
match1 = re.match(pattern, text1)
match2 = re.match(pattern, text2)

# 输出结果
if match1:
    print(f"text1: 匹配结果：{match1.group()}")
else:
    print("text1: 没有匹配结果")
if match2:
    print(f"text2: 匹配结果：{match2.group()}")
else:
    print("text2: 没有匹配结果")


# 匹配不在字符集中的字符
pattern = r"[^ol]"
matches = re.findall(pattern, text1)
print(f"匹配结果：{matches}")
```

```
text1: 匹配结果：hello
text2: 没有匹配结果
匹配结果：['h', 'e', ' ', 'w', 'r', 'd']
```

在这个例子中，`re.match` 函数会检查 `text1` 是否以 `hello` 开始，如果是，则返回一个匹配对象。对于 `text2`，由于它不是以 `hello` 开始，所以 `re.match` 不会返回任何匹配结果。对于取非的用法我们在上一小节刚刚用过，这里不再赘述。

### 4.2.9 特殊字符：`$`（锚定匹配字符串的末尾）

美元符号 `$` 用于锚定匹配字符串的末尾。当 `$` 出现在正则表达式的末尾时，它表示匹配必须发生在被搜索字符串的结束位置。也就是说，只有当被搜索的字符串以这个模式结束時，匹配才会成功。

我们直接使用 Python 进行示例说明：

```python
import re


# 示例字符串
text1 = "hello world!"
text2 = "say hello world"

# 正则表达式，用于匹配以 "world" 结尾的字符串
pattern = r"world$"

# 使用 re.match 检查匹配
match1 = re.search(pattern, text1)
match2 = re.search(pattern, text2)

# 输出结果
if match1:
    print(f"text1: 匹配结果：{match1.group()}")
else:
    print("text1: 没有匹配结果")
if match2:
    print(f"text2: 匹配结果：{match2.group()}")
else:
    print("text2: 没有匹配结果")
```

```
text1: 没有匹配结果
text2: 匹配结果：world
```

在这个例子中，`re.search` 函数用于在字符串中搜索与正则表达式匹配的子串。对于 `text2`，由于它以 "world" 结尾，所以 `world$` 将会匹配成功。而对于 `text1`，尽管它包含 "world"，但它不是字符串的结尾部分（真正结尾部分是 "world!"），因此 `world$` 不会匹配。

### 4.2.10 特殊字符：`|`（交替和 OR）

竖线 `|` 称为交替（alternation）或逻辑“或”（OR）操作符。它用于在正则表达式中分隔多个选项，允许匹配任一选项。当使用 `|` 时，正则表达式会从左到右检查每个选项，直到找到第一个匹配项。

例如，正则表达式 `cat|dog` 将匹配包含 "cat" 或 "dog" 的字符串。如果字符串中同时存在 "cat" 和 "dog"，则只会匹配第一个遇到的 "cat" 或 "dog"。

`|` 的 Python 示例：

```python
import re


# 示例字符串
text = "I have a cat and a dog."

# 正则表达式，用于匹配 "cat" 或 "dog"
pattern = r"cat|dog"

# 使用 re.findall 查找所有匹配项
matches = re.findall(pattern, text)

# 输出结果
print(f"匹配结果：{matches}")
```

在这个例子中，`re.findall` 函数用于在字符串中查找所有与正则表达式匹配的子串。由于字符串 "I have a cat and a dog." 中同时包含 "cat" 和 "dog"，所以 `re.findall` 会返回一个列表，包含两个匹配项：["cat", "dog"]。

```python
import re

# 示例字符串
text = "The quick brown fox jumps over the lazy dog."

# 正则表达式，用于匹配 "quick" 或 "lazy"
pattern = r"quick|lazy"

# 使用 re.search 查找第一个匹配项
match = re.search(pattern, text)

# 输出结果
if match:
    print(f"找到的第一个匹配结果：{match.group()}")
else:
    print("没有找到匹配结果")
```

在这个例子中，`re.search` 函数用于在字符串中搜索与正则表达式匹配的子串。由于 "quick" 出现在 "lazy" 之前，所以 `re.search` 会返回第一个匹配项 "quick"。如果想要找到所有的匹配项，可以使用 `re.findall` 方法。

---

<kbd><b>Question</b></kbd>：`|` 相当于是多个元素并列？

<kbd><b>Answer</b></kbd>：是的，在正则表达式中，`|` 符号用于表示多个元素并列，相当于多个选项的逻辑或。当我们在正则表达式中使用 `|` 时，我们实际上是在创建一个“选择”组，这个组可以匹配两个或多个不同的字符串模式中的任何一个。

---

<kbd><b>Question</b></kbd>：那我们是不是也可以使用 `[cat,dog]` 来代替 `cat|dog`？

<kbd><b>Answer</b></kbd>：不，我们不能直接使用 `[cat,dog]` 来代替 `cat|dog`。这两种语法在正则表达式中代表不同的概念。

- `[cat,dog]` 是一个字符集合，它匹配字符集中的任意一个字符。在这个例子中，它匹配字符串中的 "c"，"a"，"t"，"d"，"o"，"g" 中的任意一个。
- `cat|dog` 是一个逻辑或，它匹配字符串中的 "cat" 或 "dog"。这表示它可以匹配以 "cat" 开头或以 "dog" 结尾的字符串，或者两者都匹配。

### 4.2.11 特殊字符：`()`（分组）

括号 `()` 用于创建一个捕获组（capturing group）。捕获组允许我们将正则表达式的一部分作为一个整体来操作，并且可以将这个整体作为一个子表达式来引用。

例如，如果我们有一个正则表达式 `(ab)+`，它将匹配以下字符串：

- "abab"
- "ababab"
- "abababab"

这是因为 `(ab)` 被视为一个整体，可以重复零次或多次。

在 Python 中，我们可以使用 `re.findall` 函数来查找所有匹配的子串，并且可以使用 `match.group()` 方法来获取捕获组的匹配内容。

让我编写一个 Python 的示例：

```python
import re

# 示例字符串
text = """ab
abab
ababab
abababab
ababababab
"""

# 正则表达式，用于匹配以 "ab" 重复出现的字符串
pattern = r"(ab)+"

# 使用 re.findall 查找所有匹配项
matches = re.findall(pattern, text)

# 输出结果
print(f"匹配结果：{matches}")
```

```
匹配结果：['ab', 'ab', 'ab', 'ab', 'ab']
```

在这个例子中，我们使用正则表达式 `(ab)+` 来匹配字符串中的 "ab" 序列，并且这个序列可以重复出现。当我们使用 `re.findall` 函数在文本中查找所有匹配的子串时，它找到了以下匹配项：

- 'ab'
- 'ab'
- 'ab'
- 'ab'
- 'ab'

这些匹配项都表示字符串中连续的 "ab" 序列。由于 `(ab)+` 模式匹配的是 "ab" 序列的重复，所以 `re.findall` 返回的列表包含了所有匹配的 "ab" 序列。

## 4.3 贪婪模式和非贪婪模式

我们要把下面的字符串中的所有 html 标签都提取出来：

```html
source = '<html><head><title>Title</title>'
```

得到这样的一个列表：

```
['<html>', '<head>', '<title>', '</title>']
```

我们很容易想到使用正则表达式 `<.*>`，那我们实验一下：

```python
import re


source = '<html><head><title>Title</title>'

p = re.compile(pattern=r'<.*>')

print(p.findall(source))
```

```
['<html><head><title>Title</title>']
```

我们发现结果并不是我们想要的。这是因为在正则表达式中，`'*'`、`'+'`、`'?'` 都是贪婪的，使用它们时，会尽可能多的匹配内容，所以，`<.*>` 中的 `*`（表示任意次数的重复），一直匹配到了字符串最后的 `</title>` 里面的 `e`。

解决这个问题，就需要使用非贪婪模式，也就是在星号后面加上 `?` ，变成这样 `<.*?>`：

```python
import re


source = '<html><head><title>Title</title>'

p = re.compile(pattern=r'<.*?>')

print(p.findall(source))
```

```
['<html>', '<head>', '<title>', '</title>']
```

## 4.4 匹配某种字符类型

转义字符 `\` 后面接一些字符，表示匹配某种类型的**一个**字符，例如：

- `\d` 匹配 0-9 之间任意一个数字字符，等价于表达式 `[0-9]`
- `\D` 匹配任意一个非 0-9 数字的字符，等价于表达式 `[^0-9]`
- `\s` 是一个特殊字符，代表任意空白字符。这包括空格、制表符（`\t`）、换行符（`\n`）、回车符（`\r`）等，等价于 `[\t\n\r\f\v]`
- `\S` 匹配任意一个非空白字符，等价于表达式 `[^ \t\n\r\f\v]`
- `\w` 匹配任意一个文字字符，包括大小写字母、数字、下划线，等价于表达式 `[a-zA-Z0-9_]`
  - 默认情况也包括 Unicode 文字字符，如果指定 ASCII 码标记，则只包括 ASCII 字母
- `\W` 匹配任意一个非文字字符，等价于表达式 `[^a-zA-Z0-9_]`

💡  反斜杠 `\` 也可以用在方括号里面，比如 `[\s,.]` ，它代表了一组特殊字符的组合。这个字符集合中的每个字符都有特定的含义：

- `,` 是一个普通字符，代表逗号。
- `.` 也是一个普通字符（在 `[]` 中，特殊字符会被视为普通字符），代表点号。、

所以，`[\s,.]` 作为一个整体，表示匹配任意空白字符、逗号或点号。

在 Python 中，我们可以这样使用它：

```python
import re


text = 'Hello, World! \nThis is a test.'

pattern = r"[\s,.]"  # 匹配空白字符、,、.

matches = re.findall(pattern, text)

print(f"{matches = }")
```

```
matches = [',', ' ', ' ', '\n', ' ', ' ', ' ', '.']
```

在这个例子中，`re.findall` 函数将返回所有匹配的空白字符、逗号或点号。












# 5. 回到开头的例子

# 6. 切割字符串

# 7. 字符串替换

# 8. 练习





























# 知识来源

1. [Python编程：正则表达式](https://www.bilibili.com/video/BV1q4411y7Zh)
2. [正则表达式](https://www.byhy.net/py/lang/extra/regex/)