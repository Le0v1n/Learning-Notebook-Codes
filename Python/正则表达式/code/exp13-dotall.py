import re


# text = "Hello\nWorld"
text = """Hello
World"""
pattern = r".+"  # + 表示至少匹配一次
matches = re.findall(
    pattern=pattern, 
    string=text,
    flags=re.DOTALL
)

print(f"{matches = }")