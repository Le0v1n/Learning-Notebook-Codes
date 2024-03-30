import re

text = """张三，手机号码15945678901
李四，手机号码13945677701
王二，手机号码13845666901"""

p = r"^(?P<name>.*)，.*(?P<phone>\d{11})"

matches1 = re.findall(pattern=p, string=text, flags=re.M)
print(f"{matches1 = }")
print('='*50)

matches2 = re.finditer(p, text, re.M)

for match in matches2:
    print(f"Full match: {match.group(0)}")
    print(f"Group-name: {match.group('name')}")
    print(f"Group-phone: {match.group('phone')}")
    print('-'*50)