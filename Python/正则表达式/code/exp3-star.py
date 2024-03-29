import re


content = """苹果，是绿色的
橙子，是橙色的
香蕉，是黄色的
乌鸦，是黑色的
猴子，
"""

for item in re.findall(pattern='，.*', 
                       string=content):
    print(item)