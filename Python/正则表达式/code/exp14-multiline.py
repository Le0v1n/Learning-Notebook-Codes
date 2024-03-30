import re


# text = """Hello World
# Hello World Again!"""
text = "Hello World\nHello World Again!"
pattern = r"^Hello"  # ^ 表示开头
matches = re.findall(
    pattern=pattern, 
    string=text,
    flags=re.M
)

print(f"{matches = }")