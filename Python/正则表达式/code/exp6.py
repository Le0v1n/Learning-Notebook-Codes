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