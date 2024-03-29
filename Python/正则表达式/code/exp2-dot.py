import re


content = """苹果是绿色的
橙子是.色的
香蕉是黄色的
乌鸦是黑色的
"""

for item in re.findall(pattern='.色', 
                       string=content):
    print(item)