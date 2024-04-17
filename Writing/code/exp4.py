import torch


A = torch.randint(
    high=100, 
    size=(3, 3)
)

B = torch.randint(
    high=50, 
    size=(3, 3)
)

print(f"-----------------------------------------------")
print(f"{A = }")
print(f"{A.shape = }")
print(f"-----------------------------------------------")
print(f"{B = }")
print(f"{B.shape = }")
print(f"-----------------------------------------------")

# 向量与矩阵对应位置相加
print(f"向量与矩阵对应位置相加:\n"
      f"{A + B = }\n"
      f"{(A + B).shape = }")