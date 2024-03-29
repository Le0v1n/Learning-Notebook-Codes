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