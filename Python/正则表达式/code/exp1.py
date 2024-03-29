# 打开指定文本，获取文本中的内容
with open('Python/正则表达式/code/exp1.txt', 'r') as f:
    lines = f.readlines()

# 遍历每一行
for line in lines:
    # 查找'万/月'在字符串中的索引
    pos2 = line.find('万/月')  # 如果没有找到则返回-1
    
    if pos2 == -1:  # 说明 '万/月' 没有找到
        pos2 = line.find('万/每月')  # 查找 '万/每月' 在字符串中的索引
        
        if pos2 == -1:  # 说明 '万/每月' 也没有找到
            continue
    
    # 找到了 '万/月' 或者 '万/每月'
    idx = pos2 - 1  # 数字的末尾索引
    
    # 往前找
    while line[idx].isdigit() or line[idx] == '.':  # 如果前面的是数字或者小数点
        idx -= 1  # 继续往前找
    
    # 现在我们可以确定数字的索引范围了
    pos1 = idx + 1
    
    print(line[pos1: pos2], '万/每月')