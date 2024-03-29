import re


source = '<html><head><title>Title</title>'

p = re.compile(pattern=r'<.*?>')

print(p.findall(source))