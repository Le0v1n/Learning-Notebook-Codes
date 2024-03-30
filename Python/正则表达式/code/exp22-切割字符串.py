import re


text = "关羽; 张飞, 赵云,马超, 黄忠  李逵"

p = r"[;,\s]\s*"

matches = re.split(
    pattern=p,
    string=text
)

print(f"{matches = }")