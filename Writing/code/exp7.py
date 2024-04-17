import torch


A = torch.randint(
    high=50, 
    size=(3, 3)
)

B = torch.randint(
    high=100,
    size=(3, 3)
)

print(f"-----------------------------------------------")
print(f"{A = }")
print(f"{A.shape = }")
print(f"-----------------------------------------------")
print(f"{B = }")
print(f"{B.shape = }")
print(f"-----------------------------------------------")

print(f"[torch.mul(A, B)] 矩阵与矩阵进行对应位置元素乘法:\n"
      f"{torch.mul(A, B) = }\n"
      f"{torch.mul(A, B).shape = }")
print(f"-----------------------------------------------")
print(f"[A*B] 矩阵与矩阵进行对应位置元素乘法:\n"
      f"{A * B = }\n"
      f"{(A * B).shape = }")