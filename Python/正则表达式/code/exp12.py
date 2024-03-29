import re


text = """001-苹果价格-60
002-橙子价格-70
003-香蕉价格-80"""

pattern = r"^\d+"
matches = re.findall(pattern, text, re.M)

print(f"{matches = }")