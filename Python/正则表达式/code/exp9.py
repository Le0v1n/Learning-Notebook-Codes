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
