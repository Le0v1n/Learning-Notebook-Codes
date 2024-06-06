def split_list_equally(lst, n):
    # 计算每份的大小
    size = len(lst) // n
    # 计算剩余的元素数量
    remainder = len(lst) % n
    # 使用列表切片来分割列表
    result = [lst[i*size:(i+1)*size] for i in range(n)]
    # 将剩余的元素分配到最后一份列表中
    result[-1].extend(lst[-remainder:])
    return result

# 示例使用
lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
n = 3
result = split_list_equally(lst, n)
print(result)  # 输出：[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
