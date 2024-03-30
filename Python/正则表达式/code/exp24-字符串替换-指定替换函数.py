import re


text = """

下面是这学期要学习的课程：

<a href='https://www.bilibili.com/video/av66771949/?p=1' target='_blank'>点击这里，边看视频讲解，边学习以下内容</a>
这节讲的是牛顿第2运动定律

<a href='https://www.bilibili.com/video/av46349552/?p=125' target='_blank'>点击这里，边看视频讲解，边学习以下内容</a>
这节讲的是毕达哥拉斯公式

<a href='https://www.bilibili.com/video/av90571967/?p=33' target='_blank'>点击这里，边看视频讲解，边学习以下内容</a>
这节讲的是切割磁力线
"""

def subFunc(match):
    # match对象的group(0)返回的是整个匹配上的字符串
    src = match.group(0)
    
    # match对象的group(1)返回的是第一个group分组的内容
    number = int(match.group(1)) + 6
    dst = f"/av{number}/"
    
    print(f"💡  {src} 被替换为 {dst}")
    
    return dst


p = r"/av(\d+)/"  # 这里我们把匹配上的数字添加为一个组

text = re.sub(
    pattern=p,
    repl=subFunc,  # 注意函数不要加()，加了表示函数的调用
    string=text
)

print(f"{text = }")