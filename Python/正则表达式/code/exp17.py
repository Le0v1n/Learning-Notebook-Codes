import re


# 示例字符串
text = "I have a cat and a dog."

# 正则表达式，用于匹配 "cat" 或 "dog"
pattern = r"cat|dog"

# 使用 re.findall 查找所有匹配项
matches1 = re.findall(pattern, text)
matches2 = re.search(pattern, text)

# 输出结果
print(f"{matches1 = }")
print(f"{matches2 = }")
print(f"{matches2.group()}") if matches2 else ...
print(f"{matches2.groups()}") if matches2 else ...