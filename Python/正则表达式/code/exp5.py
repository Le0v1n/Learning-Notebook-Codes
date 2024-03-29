import re


# 匹配字符集内的任意字符
text = "abc def ghi jkl mno a b c d aa  a a a a a a sdsad sajkjclkx jsadkl dskljnsdlijewqlkjsadj lasdjlkjdwijsalkj lksajd lkasjwd"
pattern = r"[ace]"
matches = re.findall(pattern, text)
print(f"匹配结果：{matches}")

# 匹配不在字符集内的任意字符
text = "abc def ghi jkl mno a b c d aa  a a a a a a sdsad sajkjclkx jsadkl dskljnsdlijewqlkjsadj lasdjlkjdwijsalkj lksajd lkasjwd"
pattern = r"[^ace]"
matches = re.findall(pattern, text)
print(f"匹配结果：{matches}")

# 匹配字符范围
text = "abc def ghi jkl mno a b c d aa  a a a a a a sdsad sajkjclkx jsadkl dskljnsdlijewqlkjsadj lasdjlkjdwijsalkj lksajd lkasjwd"
pattern = r"[a-e]"
matches = re.findall(pattern, text)
print(f"匹配结果：{matches}")