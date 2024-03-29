import re  # re: regular expression


# 打开指定文本，获取文本中的内容
with open('Python/正则表达式/code/exp1.txt', 'r') as f:
    lines = f.readlines()
    text = ''.join(lines)  # 使用 join 方法将列表转换为字符串
    
p = re.compile('([\d.]+)万/每{0,1}月')

for one in p.findall(text):
    print(f"{one} 万/月")