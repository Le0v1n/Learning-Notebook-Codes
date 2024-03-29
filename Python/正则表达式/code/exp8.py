import re


# 示例字符串
text = "catdogactodg"


# 正则表达式，用于匹配 "cat" 或 "dog"
pattern = r"cat|dog"

# 使用 re.findall 查找所有匹配项
matches = re.findall(pattern, text)

# 输出结果
print(f"匹配结果：{matches}")