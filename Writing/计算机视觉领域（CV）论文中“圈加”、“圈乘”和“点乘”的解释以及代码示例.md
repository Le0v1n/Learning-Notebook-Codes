
# 0. 引言

我们经常在论文中看到 $\oplus$、$\otimes$ 和 $\odot$ 符号，那么有下面两个问题：

1. 这三个符号有什么作用呢？
2. 如何在论文中正确使用这三个数学符号

# 1. 两种符号的解释

## 1.1 逐元素相加：$\oplus$

$\oplus$ 在论文中表示逐元素相加，如果用两个矩阵表示，即：

$$
\left[\begin{matrix}a & b \cr c & d\end{matrix}\right]_{2\times2}+\left[\begin{matrix}e & f \cr g & h\end{matrix}\right]_{2\times2}=\left[\begin{matrix}a+e & b+f \cr c+g & d+h\end{matrix}\right]_{2\times2}
$$

从公式可以看到，$\oplus$ 表示对应元素相加，即两个矩阵的形状必须相同。

## 1.2 矩阵乘法：$\otimes$

圈乘 $\otimes$ 表示传统线性代数学的矩阵乘法，用公式即：

$$
\left[\begin{matrix}a_{11} & a_{12} & a_{13}\cr a_{21} & a_{22} & a_{23}\end{matrix}\right]_{2\times3} \times 
\left[\begin{matrix}
b_{11} & b_{12} \cr 
b_{21} & b_{22} \cr 
b_{31} & b_{32}
\end{matrix}\right]_{3\times2}
=\left[\begin{matrix}a_{11}\times b_{11} + a_{12}\times b_{21} + a_{13}\times b_{31} 
& a_{11}\times b_{12} + a_{12}\times b_{22} + a_{13}\times b_{32} 
\cr 
a_{21}\times b_{11} + a_{22}\times b_{21} + a_{23}\times b_{31} 
& a_{21}\times b_{12} + a_{22}\times b_{22} + a_{23}\times b_{32} 
\end{matrix}\right]_{2\times2}
$$

可以看到**就是普通的矩阵乘法**，要求 A 矩阵第二维度与 B 矩阵第一维度相等。

## 1.3 矩阵点乘：$\odot$

矩阵点乘 $\odot$ 表示**矩阵对应位置元素相乘**，例子如下：

$$
\left[\begin{matrix}a & b \cr c & d\end{matrix}\right]_{2\times2} \odot \left[\begin{matrix}e & f \cr g & h\end{matrix}\right]_{2\times2}=\left[\begin{matrix}a\times e & b \times f \cr c \times g & d \times h\end{matrix}\right]_{2\times2}
$$

与矩阵加法 $\oplus$ 类似，也是要求两个矩阵的维度必须相同。

# 2. 两种符号的代码表示

|名称|符号|PyTorch 代码|含义|条件|
|:-|:-:|:-|:-|:-|
|矩阵乘法(element-wise)|$\otimes$|`torch.mm()` 或 `torch.matmul()`|矩阵乘法(自动广播)|形状相同或满足广播机制|
|矩阵加法(element-wise)|$\oplus$|`+` 或 `torch.add(A, B)`|两个矩阵对应位置元素相加|形状相同或满足广播机制||
|矩阵点乘|$\odot$|`*` 或 `torch.mul(A, B)`|两个矩阵对应位置元素相乘(自动广播)|形状相同或满足广播机制|

> element-wise 表示逐元素地

# 3. PyTorch 广播机制

## 3.1 定义

在 PyTorch 中，算子 $\otimes 和 \odot$ 操作支持广播，即可以自动将两种 Tensor 中较小尺寸的 Tensor 扩展为相等大小（无需复制数据）。

## 3.2 用法

如果 Tensor `A` 和 `B` 符合广播的条件，那么可以按照下面的方式计算：
1. 如果 `A` 和 `B` 的维度不相同，用 1 来扩展维度较少的那个，使两个 Tensor 的维度一致。
2. 对于每个维度，结果的维度是 `A` 和 `B` 对应维度的最大值。即：列向量按列扩展，行向量按行扩展，使两个 Tensor 扩展为相同尺寸，然后再进行对应元素相乘。用伪代码表示为：

```python
out.shape = max(A.shape, B.shape)
```

# 4. 代码演示

## 4.1 对应位置元素相加：$\oplus$

### 4.1.1 标量（Scale）与向量（Vector）对应元素相加（自动触发广播机制）

```python
import torch


A = torch.randint(
    high=100, 
    size=(3, 3)
)

print(f"{A = }")
print(f"{A.shape = }")

# 1. 矩阵与标量对应位置相加
print(
    f"-----------------------------------------------\n"
    f"矩阵与标量对应位置相加: \n"
    f"{A + 10 = }"
    f"\n-----------------------------------------------"
)
```

```
A = tensor([[79, 20, 39],
            [31, 47,  9],
            [22, 54, 81]])
A.shape = torch.Size([3, 3])
-----------------------------------------------
矩阵与标量对应位置相加: 
A + 10 = tensor([[89, 30, 49],
                 [41, 57, 19],
                 [32, 64, 91]])
-----------------------------------------------
```

### 4.1.2 向量（Vector）与向量（Vector）对应元素相加

```python
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
```

```
A = tensor([[1],
            [2],
            [3]])
A.shape = torch.Size([3, 1])
-----------------------------------------------
B = tensor([[1, 2, 3]])
B.shape = torch.Size([1, 3])
-----------------------------------------------
向量与向量对应位置相加:
A + B = tensor([[2, 3, 4],
                [3, 4, 5],
                [4, 5, 6]])
(A + B).shape = torch.Size([3, 3])
```

### 4.1.3 向量（Vector）与矩阵（Matrix）对应元素相加

```python
import torch


A = torch.tensor(
    data=([1], [2], [3])
)

B = torch.randint(
    high=100, 
    size=(3, 3)
)

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
```

```
A = tensor([[1],
            [2],
            [3]])
A.shape = torch.Size([3, 1])
-----------------------------------------------
B = tensor([[21, 69, 84],
            [59, 46, 54],
            [68, 25, 95]])
B.shape = torch.Size([3, 3])
-----------------------------------------------
向量与矩阵对应位置相加:
A + B = tensor([[22, 70, 85],
               [61, 48, 56],
               [71, 28, 98]])
(A + B).shape = torch.Size([3, 3])
```

### 4.1.4 矩阵（Matrix）与矩阵（Matrix）对应元素相加

#### 1. 不触发广播机制

```python
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

print(f"矩阵与矩阵对应位置相加:\n"
      f"{A + B = }\n"
      f"{(A + B).shape = }")
```

```
-----------------------------------------------
A = tensor([[10, 64, 54],
            [51, 13, 99],
            [35, 62, 11]])
A.shape = torch.Size([3, 3])
-----------------------------------------------
B = tensor([[26, 20,  5],
            [40,  1, 26],
            [ 4, 20, 47]])
B.shape = torch.Size([3, 3])
-----------------------------------------------
矩阵与矩阵对应位置相加:
A + B = tensor([[ 36,  84,  59],
                [ 91,  14, 125],
                [ 39,  82,  58]])
(A + B).shape = torch.Size([3, 3])
```

#### 2. 触发广播机制

```python
import torch


A = torch.randint(
    high=100, 
    size=(3, 3)
)

B = torch.randint(
    high=50, size=(1, 3)
)

print(f"-----------------------------------------------")
print(f"{A = }")
print(f"{A.shape = }")
print(f"-----------------------------------------------")
print(f"{B = }")
print(f"{B.shape = }")
print(f"-----------------------------------------------")

# [广播机制]矩阵与矩阵对应位置相加
print(f"[广播机制] 矩阵与矩阵对应位置相加:\n"
      f"{A + B = }\n"
      f"{(A + B).shape = }")
```

```
-----------------------------------------------
A = tensor([[82, 43, 47],
            [14, 18, 57],
            [27, 45, 71]])
A.shape = torch.Size([3, 3])
-----------------------------------------------
B = tensor([[34, 40, 22]])
B.shape = torch.Size([1, 3])
-----------------------------------------------
[广播机制] 矩阵与矩阵对应位置相加:
A + B = tensor([[116,  83,  69],
                [ 48,  58,  79],
                [ 61,  85,  93]])
(A + B).shape = torch.Size([3, 3])
```

#### 3. [错误的] 触发广播机制

```python
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

print(f"[错误的广播机制] 矩阵与矩阵对应位置相加")
try:
    print(f"{A + B = }")
except Exception as e:
    print(f"A + B = {e}")
try:
    print(f"{(A + B).shape = }")
except Exception as e:
    print(f"(A + B).shape = {e}")
```

```
-----------------------------------------------
A = tensor([[82,  6, 22],
            [71, 56, 10],
            [76, 74, 80]])
A.shape = torch.Size([3, 3])
-----------------------------------------------
B = tensor([[37, 30]])
B.shape = torch.Size([1, 2])
-----------------------------------------------
[错误的广播机制] 矩阵与矩阵对应位置相加
A + B = The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1
(A + B).shape = The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1
```

## 4.2 对应元素相乘：$\odot$

```python
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
```

```
-----------------------------------------------
A = tensor([[36, 37, 31],
        [42, 15,  8],
        [ 8, 31, 18]])
A.shape = torch.Size([3, 3])
-----------------------------------------------
B = tensor([[46, 85, 34],
        [25, 14, 24],
        [89,  1, 71]])
B.shape = torch.Size([3, 3])
-----------------------------------------------
[torch.mul(A, B)] 矩阵与矩阵进行对应位置元素乘法:
torch.mul(A, B) = tensor([[1656, 3145, 1054],
        [1050,  210,  192],
        [ 712,   31, 1278]])
torch.mul(A, B).shape = torch.Size([3, 3])
-----------------------------------------------
[A*B] 矩阵与矩阵进行对应位置元素乘法:
A * B = tensor([[1656, 3145, 1054],
        [1050,  210,  192],
        [ 712,   31, 1278]])
(A * B).shape = torch.Size([3, 3])
```

> ⚠️  OBS：<font color='red'><b>程序中的 A * B 表示的是 $\odot$ 而非 $\otimes$</b></font>。

## 4.3 矩阵乘法：$\otimes$

```python
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
```

```
-----------------------------------------------
A = tensor([[14, 26, 24],
        [24, 31, 18],
        [43, 19, 14]])
A.shape = torch.Size([3, 3])
-----------------------------------------------
B = tensor([[38, 25, 26],
        [73, 75, 49],
        [17, 83, 44]])
B.shape = torch.Size([3, 3])
-----------------------------------------------
[torch.matmul(A, B)] 矩阵与矩阵进行矩阵乘法:
torch.matmul(A, B) = tensor([[2838, 4292, 2694],
        [3481, 4419, 2935],
        [3259, 3662, 2665]])
torch.matmul(A, B).shape = torch.Size([3, 3])
-----------------------------------------------
[torch.mm(A, B)] 矩阵与矩阵进行矩阵乘法:
torch.mm(A, B) = tensor([[2838, 4292, 2694],
        [3481, 4419, 2935],
        [3259, 3662, 2665]])
torch.mm(A, B).shape = torch.Size([3, 3])
```

# 5. 总结

- $\oplus$ 表示对应位置元素相加
- $\odot$ 表示对应位置元素相乘
- $\otimes$ 表示矩阵乘法（线性代数中的矩阵乘法）