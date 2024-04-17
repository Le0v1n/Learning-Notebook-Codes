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

print(f"[torch.matmul(A, B)] 矩阵与矩阵进行矩阵乘法:\n"
      f"{torch.matmul(A, B) = }\n"
      f"{torch.matmul(A, B).shape = }")
print(f"-----------------------------------------------")
print(f"[torch.mm(A, B)] 矩阵与矩阵进行矩阵乘法:\n"
      f"{torch.mm(A, B) = }\n"
      f"{torch.mm(A, B).shape = }")