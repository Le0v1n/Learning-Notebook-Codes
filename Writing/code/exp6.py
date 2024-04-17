import torch


A = torch.randint(
    high=100, 
    size=(3, 3)
)

B = torch.randint(
    high=50, 
    size=(1, 2)
)

print(f"-----------------------------------------------")
print(f"{A = }")
print(f"{A.shape = }")
print(f"-----------------------------------------------")
print(f"{B = }")
print(f"{B.shape = }")
print(f"-----------------------------------------------")

print(f"[错误的广播机制]矩阵与矩阵对应位置相加")
try:
    print(f"{A + B = }")
except Exception as e:
    print(f"A + B = {e}")
try:
    print(f"{(A + B).shape = }")
except Exception as e:
    print(f"(A + B).shape = {e}")