import re


text = 'Hello, World! \nThis is a test.'
pattern = r"[\s,.]"  # 匹配空白字符、,、.
matches = re.findall(pattern, text)

print(f"{matches = }")