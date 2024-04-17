import torch


# A = torch.tensor([1], [2], [3])  # tensor() takes 1 positional argument but 3 were given
A = torch.tensor(
    data=(1, 2, 3)
)  # A.shape = torch.Size([3])

# 为 A 增加维度
A = A.unsqueeze(-1)  # [3] -> [3, 1]

# 将 A 转置生成 B（也可以使用 A.T）
B = A.t()  # [3, 1] -> [1, 3]

print(f"{A = }")
print(f"{A.shape = }")
print(f"-----------------------------------------------")
print(f"{B = }")
print(f"{B.shape = }")
print(f"-----------------------------------------------")

# 向量与向量对应位置相加
print(f"向量与向量对应位置相加:\n"
      f"{A + B = }\n"
      f"{(A + B).shape = }")