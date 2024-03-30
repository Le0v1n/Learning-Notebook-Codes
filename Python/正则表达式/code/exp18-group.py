import re

text = """苹果，苹果是绿色的
橙子，橙子是橙色的
香蕉，香蕉是黄色的"""

p = r"^(.*)，"

matches1 = re.findall(pattern=p, string=text, flags=re.M)
print(f"{matches1 = }")
print('='*50)

matches2 = re.finditer(p, text, re.M)

for match in matches2:
    print(f"Full match: {match.group(0)}")
    print(f"Group1: {match.group(1)}")
    print('-'*50)