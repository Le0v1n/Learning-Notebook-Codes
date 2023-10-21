
# 1. hook 函数

## 1.1 hook 函数的概念

**hook 函数机制**：不改变主体，实现额外功能，像一个挂件、挂钩 → hook。

那么为什么会有 hook 函数这个机制呢？这与 PyTorch 动态图运行机制有关。在动态图运行机制中，当运算结束后，一些中间变量是会被释放掉的，比如特征图、非叶子节点的梯度。但有时候我们又想要继续关注这些中间变量，那么就可以使用 hook 函数在主体代码中提取中间变量。主体代码主要是模型的前向传播和反向传播。

简单讲，hook 函数就是不修改主体，而实现额外功能。对应到在 PyTorch 中，主体就是 `forward` 和 `backward`，而额外的功能就是对模型的变量进行操作，如：

1. “提取”特征图
2. “提取”非叶子张量的梯度
3. 修改张量梯度
4. ...

举个例子演示 hook 提取非叶子张量的梯度：

```python
import torch


# 定义钩子操作
def grad_hook(grad):
    y_grad.append(grad)
    
# 创建一个list来保存钩子获取的梯度
y_grad = list()

# 创建输入变量
x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)  # requires_grad=True表明该节点为叶子节点
y = x + 1  # 没有明确requires_grad=True，所以是非叶子节点

# 为非叶子节点y注册钩子
y.register_hook(grad_hook)  # 这是传入的是函数而非函数的调用

# y.retain_grad()  # 如果想要将 y 设置为叶子节点，可以设置 y.retain_grad()

# 计算z节点(z 也是一个非叶子节点，因为它是通过对非叶子节点 y 进行操作而得到的)
z = torch.mean(y * y)

# 反向传播
z.backward()

print(f"y.type: {type(y)}")
print(f"y.grad: {y.grad}")
print(f"y_grad: {y_grad}")
```

**结果如下**：

```
y.type: <class 'torch.Tensor'>
y.grad: None
y_grad: [tensor([[1.0000, 1.5000],
        [2.0000, 2.5000]])]
```

可以看到 `y. grad` 的值为 `None`，这是因为 `y` 是非叶子结点张量，在 `z. backward()` 完成之后，`y` 的梯度被释放掉以节省内存，但可以通过 torch. Tensor 的类方法 register_hook 将 y 的梯度提取出来。

---

这里 PyTorch 可能会报警告：

```
/root/anaconda3/envs/wsss/lib/python3.9/site-packages/torch/_tensor.py:1083: 
UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). 
If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. 
If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. 
See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at  aten/src/ATen/core/TensorBody.h:482.)
```

**翻译过来就是**：

```
/root/anaconda3/envs/wsss/lib/python3.9/site-packages/torch/_tensor.py:1083: UserWarning: 
正在访问不是叶张量的张量的 .grad 属性。 在 autograd.backward() 期间不会填充其 .grad 属性。 
如果我们确实希望为非叶张量填充 .grad 字段，请在非叶张量上使用 .retain_grad() 。 
如果我们错误地访问了非叶张量，请确保我们改为访问叶张量。 
有关更多信息，请参阅 github.com/pytorch/pytorch/pull/30531。 （在 aten/src/ATen/core/TensorBody.h:482 内部触发。）
```

这个警告通常出现在我们尝试访问不是叶子节点的张量的梯度信息时，PyTorch提醒我们梯度信息对于非叶子节点不可用。如果我们需要在非叶子节点上使用梯度信息，可以使用 `.retain_grad()` 方法来为其启用梯度信息的记录。否则，确保我们的操作是在叶子节点上执行的，以便正常访问梯度信息。

---

要在 Python 中忽略特定的警告，可以使用 `warnings` 模块。在这种情况下，我们可以通过以下方式来忽略 PyTorch 的用户警告：

```python
import warnings

# 忽略特定的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
```

上述代码会将 `UserWarning` 类别的警告从 `torch`模 块中过滤掉，使其不再显示。请注意，忽略警告是一个全局操作，因此可能会影响到我们的整个 Python 环境。确保在使用此方法时明智选择，以避免隐藏重要的警告信息。

## 1.2 PyTorch 提供的 hook 函数

1. `torch.Tensor.register_hook (Python method, in torch.Tensor)`
2. `torch.nn.Module.register_forward_hook (Python method, in torch.nn)`
3. `torch.nn.Module.register_backward_hook (Python method, in torch.nn)`
4. `torch.nn.Module.register_forward_pre_hook (Python method, in torch.nn)`

这 4 个 hook 中有一个是应用于 `tensor` 的，另外 3 个是针对 `nn.Module` 的。

### 1.2.1 Tensor.register_hook

+ **功能**：注册一个反向传播 hook 函数。因为 Tensor 只有在反向传播的时候，如果它不是叶子节点，那么它自身的梯度就会被释放掉。所以这个 hook 函数专为 Tensor 设计。

	`Tensor` 类的 `register_hook` 方法在 PyTorch 中用于注册梯度钩子（gradient hook）。梯度钩子是一种回调函数，允许我们在张量的梯度计算过程中进行自定义操作，例如捕获、修改或记录梯度信息。以下是有关 `register_hook` 方法的作用、用法和返回值的详细说明：

+ **语法**：

	```python
	def hook(grad):
	    ...
	
	tensor.register_hook(hook)
	```

+ **作用**：

	- `register_hook` 方法的主要作用是允许用户在张量的梯度计算中注册一个自定义函数，以便在反向传播期间对梯度进行操作或记录信息。

	- 这对于实现自定义梯度处理、梯度剪裁、可视化梯度信息以及梯度的修改等任务非常有用。


+ **返回值**：
	+ `register_hook` 方法的返回值是一个可用于取消梯度钩子的句柄（hook handle）。通过调用`remove()`方法，我们可以随时取消已注册的梯度钩子，以避免内存泄漏。

+ **应用场景举例**：在 hook 函数中可对梯度 grad 进行 in-place 操作，即可修改 tensor 的 grad 值。这是一个很酷的功能，例如当浅层的梯度消失时，可以对浅层的梯度乘以一定的倍数，用来增大梯度；还可以对梯度做截断，限制梯度在某一区间，防止过大的梯度对权值参数进行修改。

+ **例子**
	+ 例 1：获取中间变量 y 的梯度
	+ 例 2：利用 hook 函数将变量 x 的梯度扩大 2 倍

---

<div align=center>
	<img src=https://img-blog.csdnimg.cn/4d85c0c3230e42b7b7eba993719418e4.png
	width=80%>
</div>

```python
"""例 1：获取中间变量 a 的梯度"""
import torch
import warnings


# 忽略特定的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


# 自定义hook操作: 梯度处理或记录操作
def grad_hook(grad):
    a_grad.append(grad)


if __name__ == "__main__":
    w = torch.tensor([1.], requires_grad=True)  # 定义叶子节点
    x = torch.tensor([2.], requires_grad=True)  # 定义叶子节点
    a = torch.add(w, x)  # 非叶子节点
    b = torch.add(w, 1)  # 非叶子节点
    y = torch.mul(a, b)  # 非叶子节点

    # 存放梯度
    a_grad = []
    
    # 注册梯度钩子
    handle = a.register_hook(grad_hook)
    
    # 反向传播
    y.backward()

    # 查看梯度
    print(f"w.grad: {w.grad}")  # tensor([5.])
    print(f"x.grad: {x.grad}")  # tensor([2.])
    print(f"a.grad: {a.grad}")  # None
    print(f"b.grad: {b.grad}")  # None
    print(f"y.grad: {y.grad}")  # None
    print(f"a_grad: {a_grad}")  # [tensor([2.])]
    print(f"a_grad[0]: {a_grad[0]}")  # tensor([2.])
    
    # 取消钩子，避免内存泄漏
    handle.remove()
```

```
w.grad: tensor([5.])
x.grad: tensor([2.])
a.grad: None
b.grad: None
y.grad: None
a_grad: [tensor([2.])]
a_grad[0]: tensor([2.])
```

在上面的示例中，`grad_hook `函数被注册到 `tensor` 张量上，并在反向传播时被触发，将其梯度存储到 `a_grad` 列表中，从而保留其梯度信息。

```python
"""例 2：利用 hook 函数将变量 x 的梯度扩大 2 倍"""
import torch
import warnings


# 忽略特定的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# 定义钩子操作
def grad_hook(grad):
    grad *= 2
    return grad


# 创建输入变量
x = torch.tensor([2., 2., 2., 2.], requires_grad=True)  # requires_grad=True表明该节点为叶子节点
y = torch.pow(x, 2)  # 没有明确requires_grad=True，所以是非叶子节点
z = torch.mean(y)  # 对非叶子节点 y 进行操作，所以 z 也不是叶子节点

# 为非叶子节点 y 注册钩子, 返回值为 Handler
handler = x.register_hook(grad_hook)

# 反向传播
z.backward()

print(f"x.grad: {x.grad}")

# 取消梯度钩子
handler.remove()
```

```
x.grad: tensor([2., 2., 2., 2.]
```

原 x 的梯度为` tensor([1., 1., 1., 1. ])`，经 grad_hook 操作后，梯度为 `tensor([2., 2., 2., 2. ])`。

总之，`register_hook`方法使我们可以在PyTorch中自定义梯度处理逻辑，为梯度计算添加额外的控制和功能。

### 1.2.2 nn.Module.register_forward_hook

`nn.Module.register_forward_hook` 是 PyTorch 中的一个方法，用于在神经网络模块（`nn.Module`）的前向传播过程中注册回调函数（hook）。这允许我们捕获模块的输入和输出，以便进行自定义操作或记录信息。

+ **功能**

	1. **监控模块的输入和输出：** 通过注册前向传播钩子，我们可以捕获神经网络模块的输入和输出。这对于了解模块如何处理数据非常有用，以及监视中间状态。


	2. **记录中间状态：** 我们可以在前向传播过程中捕获模块的中间状态，例如隐藏层的输出。这对于可视化中间特征、模型调试和模型解释性非常有帮助。


	3. **自定义操作：** 我们可以在前向传播钩子中执行自定义操作，例如对模块的输出进行修改，添加噪声，或者进行其他自定义处理。


	4. **特定任务的应用：** 在某些任务中，前向传播钩子可以用于实现特定的功能，例如对激活函数的输出进行特定的处理，或者将中间特征传递给其他模块。


+ **语法**：

	```python
	def hook(module, input, output) -> None:
	    ...
	
	model/layer.register_forward_hook(hook)
	```

	+ `module`：当前网络层

	+ `input`：当前网络层输入数据

	+ `output`：当前网络层输出数据

	> 注意不能修改 input 和 output

+ **应用场景举例**：用于提取特征图
	假设网络由卷积层 `conv1` 和池化层 `pool1` 构成，输入一张 $4\times4$ 的图片，现采用 `forward_hook` 获取 `module` 的 `conv1` 之后的 feature maps，示意图如下：

	<div align=center>
		<img src=https://img-blog.csdnimg.cn/dfbe4bf530f64068821dc43ded5960d1.png
		width=100%>
	</div>	

	```python
	import torch
	import torch.nn as nn
	import warnings
	
	
	# 忽略特定的警告
	warnings.filterwarnings("ignore", category=UserWarning, module="torch")
	
	
	class CustomModel(nn.Module):
	    def __init__(self):
	        super(CustomModel, self).__init__()
	        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3)
	        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
	
	    def forward(self, x):
	        x = self.conv1(x)
	        x = self.pool1(x)
	        return x
	
	
	def forward_hook(module, input, output):
	    outputs_fmap_list.append(output)
	    inputs_fmap_list.append(input)
	
	
	if __name__ == "__main__":
	    # 初始化网络
	    model = CustomModel()
	    model.conv1.weight[0].data.fill_(1)
	    model.conv1.weight[1].data.fill_(2)
	    model.conv1.bias.data.zero_()
	
	    # 定义保存输入、输出feature maps的list
	    outputs_fmap_list = list()
	    inputs_fmap_list = list()
	
	    # 注册hook
	    model.conv1.register_forward_hook(forward_hook)
	
	    # 模型前向推理
	    dummy_img = torch.ones((1, 1, 4, 4))   # batch size * channel * H * W
	    output = model(dummy_img)
	
	    # 观察
	    print(f"output shape: \n\t{output.shape}")
	    print(f"output value: \n\t{output}")
	    print("---" * 30)
	    print(f"feature maps shape: \n\t{outputs_fmap_list[0].shape}")
	    print(f"output value: \n\t{outputs_fmap_list[0]}")
	    print("---" * 30)
	    print(f"input shape: \n\t{inputs_fmap_list[0][0].shape}")
	    print(f"input value: \n\t{inputs_fmap_list[0]}")
	```

	```
	output shape: 
	        torch.Size([1, 2, 1, 1])
	output value: 
	        tensor([[[[ 9.]],
	
	         [[18.]]]], grad_fn=<MaxPool2DWithIndicesBackward0>)
	------------------------------------------------------------------------------------------
	feature maps shape: 
	        torch.Size([1, 2, 2, 2])
	output value: 
	        tensor([[[[ 9.,  9.],
	                  [ 9.,  9.]],
	                  
	                 [[18., 18.],
	                  [18., 18.]]]], grad_fn=<ConvolutionBackward0>)
	------------------------------------------------------------------------------------------
	input shape: 
	        torch.Size([1, 1, 4, 4])
	input value: 
	        (tensor([[[[1., 1., 1., 1.],
	                   [1., 1., 1., 1.],
	             	   [1., 1., 1., 1.],
	          		   [1., 1., 1., 1.]]]]),)
	```

	首先初始化一个网络，卷积层有两个卷积核，权值分别为全 1 和全 2，bias 设置为 0，池化层采用 $2\times2$ 的最大池化。
	在进行 `forward` 之前对 module 的 `conv1` 注册了 `forward_hook` 函数，然后执行前向传播（`output = model(dummy_img)`），当前向传播完成后，`outputs_fmap_list` 列表中的第一个元素就是 `conv1` 层输出的特征图了。
	这里注意观察 `forward_hook` 函数有 `input` 和 `output` 两个变量，特征图是 `output` 这个变量，而 `input` 是 `conv1` 层的输入数据，`conv1` 层的输入是一个 tuple 的形式。

---

**下面剖析一下 module 是怎么样调用 hook 函数的呢**？

1. `model` 是一个 `module` 类，对 `module` 执行 `module(input)` （`output = model(dummy_img)`）是会调用 `module.call`

2. 而 `module.__call__` 执行流程如下：

	```python
	def __call__(self, *input, **kwargs):
	    for hook in self._forward_pre_hooks.values():
	        hook(self, input)
	    if torch._C._get_tracing_state():
	        result = self._slow_forward(*input, **kwargs)
	    else:
	        result = self.forward(*input, **kwargs)
	    for hook in self._forward_hooks.values():
	        hook_result = hook(self, input, result)
	        if hook_result is not None:
	            raise RuntimeError(
	                "forward hooks should never return any values, but '{}'"
	                "didn't return None".format(hook))
	...
	```

	+ 首先判断 `module`(即 `model`)是否有 `forward_pre_hook`（在执行 `forward ` 之前的 `hook`）；

	+ 然后执行 `forward`；

	+ `forward` 结束之后才到 `forward_hook`

		但是这里注意了，现在执行的是 `model.call`，我们组成的 hook 是在 module 的 `model.conv1` 中，
		所以第 2 个跳转是在 `model.__call__` 的 `result = self.forward(*input, **kwargs)`

3. `model.forward`

	```python
	def forward(self, x):
	    x = self.conv1(x)
	    x = self.pool1(x)
	    return x
	```

	在 `model.forward` 中，首先执行 `self.conv1(x)`, 而 `conv1` 是一个 `nn.Conv2d`（也是一个 module 类）。在 step.1 中有说到，对 module 执行 `module(input)` 是会调用 `module.call`

4. `nn.Conv2d.call`

	在 `nn.Conv2d.__call__` 中与 step.2 中说到的流程是一样的：

	```python
	def __call__(self, *input, **kwargs):
	    for hook in self._forward_pre_hooks.values():
	        hook(self, input)
	    if torch._C._get_tracing_state():
	        result = self._slow_forward(*input, **kwargs)
	    else:
	        result = self.forward(*input, **kwargs)
	    for hook in self._forward_hooks.values():
	        hook_result = hook(self, input, result)
	        if hook_result is not None:
	            raise RuntimeError(
	                "forward hooks should never return any values, but '{}'"
	                "didn't return None".format(hook))
	```

	在这里终于要执行我们注册的 `forward_hook` 函数了，就在 `hook_result = hook(self, input, result)` 这里。此时我们需要注意两点：

	1. `hook_result = hook(self, input, result)` 中的 `input` 和 `result` 的值不可以修改！
		这里的 `input` 对应 `forward_hook` 函数中的 input；result 对应 `forward_hook` 函数中的 output，在 `conv1` 中，input 就是该层的输入数据，result 就是经过 `conv1` 层操作之后的输出特征图。虽然可以通过 hook 来对这些数据操作，但是不能修改这些值，否则会破坏模型的计算。

	2. 注册的 hook 函数是不能带返回值的，否则抛出异常

		```python
		if hook_result is not None:
			raise RuntimeError
		```

**总结一下调用流程**：

```python
model(dummy_img) --> model.call:
    result = self.forward(*input, **kwargs)
--> model.forward: 
    x = self.conv1(x)
--> conv1.call:
    hook_result = hook(self, input, result)  # hook就是我们注册的forward_hook函数了
```

### 1.2.3 nn.Module.register_forward_pre_hook

`nn.Module.register_forward_pre_hook` 是 PyTorch 中的一个方法，用于在神经网络模块的前向传播过程之前注册回调函数（hook）。这允许我们在模块的前向传播开始之前进行自定义操作或记录信息。以下是关于该方法的详细信息：

+ **定义**：`register_forward_pre_hook` 方法用于注册前向传播预钩子（forward pre-hook），允许我们在模块的前向传播之前执行操作。

+ **功能**：

	- 前向传播预钩子允许我们监视模块的输入，但在模块进行前向传播计算之前执行操作。

	- 这对于在前向传播过程中进行修改、监视输入或记录信息非常有用。


+ **语法**：

	```python
	def hook(module, input) -> None:
	    ...
	
	model/layer.register_forward_pre_hook(hook)
	```

	`hook` 是用户自定义的函数，该函数将在模块的前向传播之前执行。该函数接受两个参数：`module` 和 `input`，分别表示神经网络模块和模块的输入。

**示例**：下面是一个示例，演示如何使用`register_forward_pre_hook`方法：

```python
import torch
import torch.nn as nn
import warnings


# 忽略特定的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# 定义前向传播预钩子函数
def forward_pre_hook(module, input):
    print(f"Module: {module.__class__.__name__}")
    print(f"Input: {input}")

# 创建一个神经网络模块
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(x)

# 创建模块实例
model = MyModule()

# 创建一个输入
input_data = torch.randn(1, 3, requires_grad=True)

# 注册前向传播预钩子
hook_handle = model.fc.register_forward_pre_hook(forward_pre_hook)

# 执行前向传播
output = model(input_data)
```

```
Module: Linear
Input: (tensor([[-0.2644,  1.9462,  2.2998]], requires_grad=True),)
```

在这个示例中，我们在前向传播预钩子中捕获了模块的输入信息，并在前向传播过程开始之前执行了自定义操作。这使我们可以在模块执行前进行操作或记录输入，然后模块继续执行正常的前向传播计算。

### 1.2.4 nn.Module.register_backward_hook

`nn.Module` 中的 `register_backward_hook` 方法允许我们在 PyTorch 模型的反向传播（backpropagation）过程中注册自定义的回调函数，以便在计算梯度时执行额外的操作。这对于监视和修改梯度、进行梯度分析或执行其他自定义操作非常有用。

+ **语法**：

	```python
	def hook(module, grad_input, grad_output) -> Tensor or None:
		...
	    
	model/layer.register_backward_hook(hook)
	```

+ **参数说明**：
	+ `module`：表示模型中的层或模块
	+ `grad_input`：一个包含输入梯度的元组(tuple)
	+ `grad_output`：一个包含输出梯度的元组(tuple)

回调函数通常用于执行一些特定操作，例如记录、分析或修改梯度。我们可以在模型的不同层上注册不同的回调函数，以便在需要时针对不同的层执行不同的操作。

+ **应用场景举例**：提取特征图的梯度
	采用 `register_backward_hook` 实现特征图梯度的提取，并结合 Grad-CAM（基于类梯度的类激活图可视化）方法对卷积神经网络的学习模式进行可视化。

以下是一个示例，演示如何使用 `register_backward_hook`：

```python
import torch
import torch.nn as nn
import warnings


# 忽略特定的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# 自定义的回调函数
def hook(module, grad_input, grad_output):
    # 打印梯度信息
    print(f"Module: \n{module}\n")
    print(f"Input Gradient: \n{grad_input}\n")
    print(f"Output Gradient: \n{grad_output}\n")

# 创建模型
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        return x

if __name__ == "__main__":
    model = CustomModel()
    loss_fn = nn.MSELoss()

    # 在模型的某一层上注册回调函数
    handler = model.conv1.register_backward_hook(hook)

    # 前向传播和反向传播
    dummy_img = torch.ones((1, 1, 4, 4))
    label = torch.randint(0, 10, size=[1, 2], dtype=torch.float)
    output = model(dummy_img)
    loss = loss_fn(output, label)  # 将输出转化为标量值
    loss.backward()  # 对损失进行反向传播


    # 注销回调函数
    handler.remove()
```

```
Module: Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1))

Input Gradient: 
(None, tensor([[[[-2.7914, -2.7914, -2.7914],
          		 [-2.7914, -2.7914, -2.7914],
          		 [-2.7914, -2.7914, -2.7914]]],


        		[[[-1.2109, -1.2109, -1.2109],
          		  [-1.2109, -1.2109, -1.2109],
          		  [-1.2109, -1.2109, -1.2109]]]]), tensor([-2.7914, -1.2109]))

Output Gradient: 
(tensor([[[[-2.7914,  0.0000],
           [ 0.0000,  0.0000]],

          [[-1.2109,  0.0000],
           [ 0.0000,  0.0000]]]]),)
```

在这个示例中，我们创建了一个自定义的回调函数 `custom_backward_hook`，并将其注册到模型的某一层上。然后，我们进行前向传播和反传播，并在回调函数中打印梯度信息。最后，我们注销了回调函数，以确保它不再在后续的反向传播中执行。

通过使用 `register_backward_hook`，我们可以监视和操作梯度，进行梯度分析，或执行其他与梯度相关的自定义操作，这对于模型的调试和优化非常有帮助。

# 2. hook 函数与特征图提取



```python
"""
    使用hook函数可视化特征图
"""
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torchvision.utils as vutils
import matplotlib.pyplot as plt


def set_seed(seed):
    # 设置Python内置的随机数生成器的种子
    random.seed(seed)

    # 设置NumPy的随机数生成器的种子
    np.random.seed(seed)

    # 设置PyTorch的随机数生成器的种子（如果使用了PyTorch）
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(1)  # 设置随机数种子
    # 实例化 Tensorboard
    writer = SummaryWriter(comment="test_your_comment", filename_suffix="_test_your_filename_suffix")

    # 读取数据并进行预处理
    image_path = "lena.png"
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]

    normalization = transforms.Normalize(MEAN, STD)
    image_transforms = transforms.Compose([transforms.Resize(size=(224, 224)),
                                           transforms.ToTensor(),
                                           normalization])  # 标准化一定要在ToTensor之后进行

    # 使用 Pillow 读取图片
    image_pillow = Image.open(fp=image_path).convert('RGB')

    # 对图片进行预处理
    if image_transforms:
        image_tensor = image_transforms(image_pillow)

    # 添加 Batch 维度
    image_tensor = torch.unsqueeze(input=image_tensor, dim=0)  # [C, H, W] -> [B, C, H, W]

    # 创建模型
    alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)  # <=> pretrained=True

    # 注册hook
    fmap_dict = dict()  # 存放所有卷积层的特征图
    for name, sub_module in alexnet.named_modules():
        """
            name: features.0
            sub_module: Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        """
        if isinstance(sub_module, nn.Conv2d):  # 判断是否为卷积层，如果是则注册hook
            key_name = str(sub_module.weight.shape)  # torch.Size([64, 3, 11, 11])
            fmap_dict.setdefault(key_name, list())
            layer_name, index = name.split('.')  # 'features', 0

            def hook_func(m, i, o):  # m: module; i: input; o: output
                key_name = str(m.weight.shape)
                fmap_dict[key_name].append(o)

            # 给 nn.Conv2d层 添加hook函数
            # alexnet._modules[layer_name]._modules[index] -> Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            alexnet._modules[layer_name]._modules[index].register_forward_hook(hook_func)

    # forward（在执行模型时会自动执行hook函数从而往fmap_dict字典中存放输出特征图）
    output = alexnet(image_tensor)

    # 添加图像
    for layer_name, fmap_list in fmap_dict.items():
        # layer_name: torch.Size([1, 3, 224, 224])
        # len(fmap_list): 1 -> shape: [B, C, H, W]
        fmap = fmap_list[0]  # 去掉[]
        fmap.transpose_(0, 1)  # [B, C, H, W] -> [C, B, H, W]

        nrow = int(np.sqrt(fmap.shape[0]))  # 开根号获取行号
        fmap_grid = vutils.make_grid(fmap, normalize=True, scale_each=True, nrow=nrow)  # [3, 458, 458]

        # 将结果存放到 Tensorboard 中
        writer.add_image(tag=f"feature map in {layer_name}", img_tensor=fmap_grid, global_step=1)

        # 也可以将结果直接用 Matplotlib 读取
        # 创建一个图像窗口
        plt.figure(figsize=(8, 8))

        # 使用imshow函数显示 grid_image
        plt.imshow(vutils.make_grid(fmap_grid, normalize=True).permute(1, 2, 0))  # 注意permute的用法，将通道维度移到最后
        plt.axis('off')  # 不显示坐标轴

        # 显示图像
        plt.savefig(f"feature map_in_{layer_name}.png")
```

<br></br>

<div align=center>

<font size=16><b>程序运行结果</b></font>

</div>

<div align=center>
	<img src=https://img-blog.csdnimg.cn/946893e210164ae2ab62141694c9851d.png
	width=100%>
	<p>
		<b>
			feature map_in_torch.Size([64, 3, 11, 11]).png
		</b>
	</p>
</div>

---

<div align=center>
	<img src=https://img-blog.csdnimg.cn/9ae314c358c044fa89493198755596ca.png
	width=100%>
	<p>
		<b>
			feature map_in_torch.Size([192, 64, 5, 5]).png
		</b>
	</p>
</div>

---

<div align=center>
	<img src=https://img-blog.csdnimg.cn/61195d6b0551404e9fe5656d46284b0e.png
	width=100%>
	<p>
		<b>
			feature map_in_torch.Size([256, 256, 3, 3]).png
		</b>
	</p>
</div>

---

<div align=center>
	<img src=https://img-blog.csdnimg.cn/b6cd7607696f499788d9195a90234599.png
	width=100%>
	<p>
		<b>
			feature map_in_torch.Size([256, 384, 3, 3]).png
		</b>
	</p>
</div>

---

<div align=center>
	<img src=https://img-blog.csdnimg.cn/0edc5b72050942619b600c0e3042ad96.png
	width=100%>
	<p>
		<b>
			feature map_in_torch.Size([384, 192, 3, 3]).png
		</b>
	</p>
</div>

## 【拓展知识】1. `dict.setdefault(key, default)` 的作用

`dict.setdefault(key, default)` 是 Python 字典（`dict`）的一个方法，用于在字典中设置键的默认值。如果字典中存在指定的键 `key`，则该方法返回与该键关联的值。如果字典中不存在指定的键 `key`，则将该键添加到字典，并将其值设置为 `default`，然后返回 `default`。这个方法允许我们为字典中的键设置默认值，以避免在访问不存在的键时引发 KeyError 异常。

## 【拓展知识】2. `alexnet._modules` 和 `alexnet.module` 有什么区别？

在 PyTorch 中，`alexnet._modules` 和 `alexnet.module` 表示不同的内容：

1. `alexnet._modules`：

	- `alexnet._modules` 是一个字典，其中包含了 AlexNet 模型的各个子模块。每个子模块是通过命名方式存储的，可以通过字典的键来访问。这些子模块包括卷积层、全连接层、池化层等。我们可以使用这种方式来查看和访问 AlexNet 的不同组件。

	+ **例如**：
	
		```python
		import torchvision.models as models
		alexnet = models.alexnet()
		print(alexnet._modules)
		```
	
		这将显示一个包含 AlexNet 各个子模块的字典：
	
		```
		OrderedDict([('features', Sequential(
		  (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
		  (1): ReLU(inplace=True)
		  (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
		  (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
		  (4): ReLU(inplace=True)
		  (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
		  (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		  (7): ReLU(inplace=True)
		  (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		  (9): ReLU(inplace=True)
		  (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		  (11): ReLU(inplace=True)
		  (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
		)), ('avgpool', AdaptiveAvgPool2d(output_size=(6, 6))), ('classifier', Sequential(
		  (0): Dropout(p=0.5, inplace=False)
		  (1): Linear(in_features=9216, out_features=4096, bias=True)
		  (2): ReLU(inplace=True)
		  (3): Dropout(p=0.5, inplace=False)
		  (4): Linear(in_features=4096, out_features=4096, bias=True)
		  (5): ReLU(inplace=True)
		  (6): Linear(in_features=4096, out_features=1000, bias=True)
		))])
		```

2. `alexnet.module`：

	+ `alexnet.module` 通常是一个属性，它用于指代整个 AlexNet 模型，特别是在分布式训练或多 GPU 训练时。在这种情况下，`alexnet.module` 是模型的顶层包装，而实际的 AlexNet 模型位于其内部。这种方式允许将模型移动到不同的 GPU 上以进行并行训练。

	+ **例如**：

		```python
		import torchvision.models as models
		alexnet = models.alexnet()
		print(alexnet.modules)
		```

		**结果如下**：

		```
		<bound method Module.modules of AlexNet(
		  (features): Sequential(
		    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
		    (1): ReLU(inplace=True)
		    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
		    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
		    (4): ReLU(inplace=True)
		    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
		    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		    (7): ReLU(inplace=True)
		    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		    (9): ReLU(inplace=True)
		    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		    (11): ReLU(inplace=True)
		    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
		  )
		  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
		  (classifier): Sequential(
		    (0): Dropout(p=0.5, inplace=False)
		    (1): Linear(in_features=9216, out_features=4096, bias=True)
		    (2): ReLU(inplace=True)
		    (3): Dropout(p=0.5, inplace=False)
		    (4): Linear(in_features=4096, out_features=4096, bias=True)
		    (5): ReLU(inplace=True)
		    (6): Linear(in_features=4096, out_features=1000, bias=True)
		  )
		)>
		```

		还有一种常用的用法：

		```python
		import torch.nn as nn
		import torch.nn.parallel
		import torchvision.models as models
		
		alexnet = models.alexnet()
		# 如果使用多GPU训练，通常有一个外部的包装模块
		alexnet = nn.DataParallel(alexnet)
		
		# 在这种情况下，实际的AlexNet模型在alexnet.module中
		actual_alexnet = alexnet.module
		```

总之，`alexnet._modules` 包含 AlexNet 的各个子模块，而 `alexnet.module` 是通常用于多 GPU 训练时的模型包装。根据我们的具体用途，我们可以选择使用其中一个。如果我们只是想查看 AlexNet 的子模块，`alexnet._modules` 是更合适的选项。如果我们需要进行多 GPU 训练，那么 `alexnet.module` 可能会更有用。

## 【拓展知识】3. `torchvision.utils.make_grid` 的作用、参数、返回值

`torchvision.utils.make_grid` 是 PyTorch 的 torchvision 库中的一个函数，用于将多个图像合并到一个大的网格中，以便更容易进行可视化和展示。这个函数通常用于可视化深度学习模型的输出或训练数据。

以下是 `make_grid` 函数的参数和返回值：

+ **参数：**

	+ `tensor`（张量）: 包含要合并为网格的图像的输入张量。通常，这是一个形状为 `(batch_size, channels, height, width)` 的张量，其中 `batch_size` 表示批处理中的图像数量，`channels` 表示通道数，`height` 和 `width` 表示每个图像的高度和宽度。

	- `nrow`（可选）: 指定每行显示的图像数量，默认为 8。


	- `padding`（可选）: 每个图像之间的填充像素数，默认为 2。


	- `normalize`（可选）: 一个布尔值，指示是否对输入张量进行归一化以在 0 到 1 范围内显示，默认为 False。如果设置为 True，输入张量将被归一化到 [0, 1]，以便在网格中更容易可视化。


	- `range`（可选）: 一个长度为 2 的元组，指定张量的数据范围，例如 `(min_value, max_value)`。如果 `normalize` 设置为 True，此参数可以用来指定数据的归一化范围。


	- `scale_each`（可选）: 一个布尔值，指示是否为每个图像分别进行缩放以适应其范围，如果设置为 True，则每个图像的范围将独立计算，默认为 False。


	- `pad_value`（可选）: 填充值的颜色，通常是一个长度为 3 的 RGB 颜色元组，默认为 0，表示黑色。


+ **返回值：**
	+ `grid_image`（张量）: 一个合并了输入张量中所有图像的张量，形状为 `(C, H, W)`，其中 `C` 是通道数，`H` 是网格的高度，`W` 是网格的宽度。通常，这个张量可以被直接用于可视化或保存为图像文件。

**示例用法**：

```python
import torchvision.utils as vutils
import torch

# 假设有一个存储图像的张量 images
grid_image = vutils.make_grid(images, nrow=4, padding=2, normalize=True)
```

在上述示例中，`make_grid` 将 `images` 张量中的图像合并为一个网格，每行显示 4 个图像，带有 2 像素的填充，并进行归一化以便可视化。合并后的图像存储在 `grid_image` 变量中。

## 【拓展知识】4. `writer.add_image` 的作用、参数和返回值

`writer.add_image` 是 PyTorch 的 `SummaryWriter` 对象中的一个方法，用于将图像添加到 TensorBoard 记录中，以便在 TensorBoard 中进行可视化、监视和分析图像数据。以下是 `writer.add_image` 方法的参数和返回值：

+ **作用：**
	- `writer.add_image` 用于记录图像数据，以便在 TensorBoard 中进行可视化。这对于可视化深度学习中的图像数据、模型输出或数据处理步骤非常有用。


+ **参数：**

	- `tag`（字符串）：标签或标识，用于标识记录的图像。通常，这是一个描述图像的字符串，以便在 TensorBoard 中标识和组织图像记录。

	- `img_tensor`（张量）：包含要记录的图像数据的张量。通常，这是一个形状为 `(C, H, W)` 的三维张量，其中 `C` 表示通道数，`H` 表示高度，`W` 表示宽度。`img_tensor` 包含要记录的图像像素值。

	- `global_step`（整数，可选）：表示记录的全局步骤或迭代次数，用于在 TensorBoard 中对齐不同记录的图像。如果不提供，TensorBoard 将使用自动增长的步骤。 —— 在模型训练中，一般就是 Epoch


+ **返回值：**
	- `None`：`writer.add_image` 方法没有返回值。它主要用于将图像数据记录到 TensorBoard 中，以供可视化和分析使用。


+ **示例用法**：

	```python
	from torch.utils.tensorboard import SummaryWriter
	import torch
	
	
	# 创建一个 SummaryWriter 对象，用于记录数据到 TensorBoard
	writer = SummaryWriter()
	
	# 假设我们已经创建了图像张量 img_tensor 和一个全局步骤 global_step
	# 将图像记录到 TensorBoard 中
	writer.add_image("Sample Image", img_tensor, global_step)
	
	# 关闭 SummaryWriter
	writer.close()
	```

	在上述示例中，`add_image` 方法将 `img_tensor` 添加到 TensorBoard 记录中，并使用 "Sample Image" 标签标识图像。我们可以在 TensorBoard 中查看和分析这些图像记录。`global_step` 参数用于指定记录的全局步骤（一般我们认为 `global_step` 就是 Epoch 或 Iteration），以便在 TensorBoard 中将图像与其他记录对齐。

# 3. CAM（Class Activation Map，类激活图）& Grad-CAM

在 1.2.4 中介绍了 `nn.module.register_backward_hook`，其实这个钩子函数经常用于 CAM 中，从而获取 HeatMap。

<div align=center>
	<img src=https://img-blog.csdnimg.cn/908af26b605b4395a7a7fe5792018fb9.png
	width=100%>
</div>

[CAM](https://arxiv.org/abs/1512.04150) 有一个缺点：网络最终输出的部分必须要有 GAP（Global Average Pooling，全局平均池化）才能得到不同特征图的权值，从而获取 HeatMap，因此要想使用 CAM，必须对网络结构进行修改，所以它的适用范围不是那么广。针对 CAM 存在的缺点，新的方法被提出：[Grad-CAM](https://arxiv.org/abs/1610.02391)。

<div align=center>
	<img src=https://img-blog.csdnimg.cn/190da2bd6f1f4c3f8371034fb27b3a37.png
	width=100%>
	<p>
		<b>
			Fig 2：Grad-CAM 概述：给定一张图像和感兴趣的类别（例如，“虎猫”或任何其他可微分输出类型）作为输入，我们通过模型的 CNN 部分进行前向传播，然后通过特定于任务的计算来获得该类别的原始分数。除了期望的类别（虎猫），梯度对所有其他类别都设置为零，而对期望的类别设置为 1。然后，将这个信号反向传播到感兴趣的修正卷积特征图，将它们组合起来计算粗略的 Grad-CAM 定位（蓝色热图），表示模型需要查看哪些部分来做出特定的决策。最后，我们将热图与引导反向传播相乘，以获得高分辨率和概念特定的 Guided Grad-CAM 可视化。
		</b>
	</p>
</div>

---

**CAM 有两个重点**：

1. 特征图；
2. 特征图对应的权重

而 Grad-CAM 使用梯度作为特征图的权重。

---

下面采用一个 [LeNet-5](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) 演示 `backward_hook` 在 Grad-CAM 中的应用。**代码流程如下**：

1. 创建网络 `net`；
2. 注册 `forward_hook` 函数用于提取最后一层特征图；
3. 注册 `backward_hook` 函数用于提取类向量（one-hot）关于特征图的梯度；
4. 对特征图的梯度进行求均值，并对特征图进行加权；
5. 可视化 HeatMap。

```python
"""
通过实现 Grad-CAM 学习 module 中的 forward_hook 和 backward_hook 函数
"""
import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
    return img


def img_preprocess(img_in):
    """
    读取图片，转为模型可读的形式
    :param img_in: ndarray, [H, W, C]
    :return: PIL.image
    """
    img = img_in.copy()
    img = cv2.resize(img, (32, 32))
    img = img[:, :, ::-1]   # BGR --> RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img_input = img_transform(img, transform)
    return img_input


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


def farward_hook(m, i, o):
    fmap_block.append(o)


def show_cam_on_image(img, mask, out_dir):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    path_cam_img = os.path.join(out_dir, "cam.jpg")
    path_raw_img = os.path.join(out_dir, "raw.jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    cv2.imwrite(path_raw_img, np.uint8(255 * img))


def comp_class_vec(ouput_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, 10).scatter_(1, index, 1)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605

    return class_vec


def gen_cam(feature_map, grads):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

    weights = np.mean(grads, axis=(1, 2))  #

    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (32, 32))
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam


if __name__ == '__main__':

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path_img = os.path.join("cam_img", "test_img_8.png")
    path_net = os.path.join("net_params_72p.pkl")
    output_dir = os.path.join("Result", "backward_hook_cam")

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    fmap_block = list()
    grad_block = list()
    print(path_img)

    # 图片读取；网络加载
    img = cv2.imread(path_img, 1)  # H*W*C
    img_input = img_preprocess(img)
    net = Net()
    net.load_state_dict(torch.load(path_net))

    # 注册hook
    net.conv2.register_forward_hook(farward_hook)
    net.conv2.register_backward_hook(backward_hook)

    # forward
    output = net(img_input)
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))

    # backward
    net.zero_grad()
    class_loss = comp_class_vec(output)
    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()
    cam = gen_cam(fmap, grads_val)

    # 保存cam图片
    img_show = np.float32(cv2.resize(img, (32, 32))) / 255
    show_cam_on_image(img_show, cam, output_dir)
```

需要注意的是在 `backward_hook` 函数中，`grad_out` 是一个 tuple 类型的，要取得特征图的梯度需要这样 `grad_block. append(grad_out[0]. detach())`

这里对 3 张飞机的图片进行观察 HeatMap，如下图所示，第一行是原图，第二行是叠加了 HeatMap 的图片。

<div align=center>
	<img src=https://img-blog.csdnimg.cn/20fb8be1e6434d2fa0ac000a8b96a857.png
	width=80%>
</div>

这里发现一个有意思的现象，模型将图片判为飞机的依据是蓝天，而不是飞机（图 1 \~ 图3）。那么我们喂给模型一张纯天蓝色的图片，模型会判为什么呢？如图 4 所示，发现模型将该图片判定为飞机。

从这里发现，虽然该模型能将飞机正确分类，但是它学到的却不是飞机的特征。这导致模型的泛化性能大打折扣，里我们可以考虑采用 trick 让模型强制的学习到飞机而不是常与飞机一同出现的蓝天，或者是调整数据。

---

对于图 4 疑问：HeatMap 蓝色区域是否对图像完全不起作用呢？是否仅仅通过红色区域就可以对图像进行判别呢？

<div align=center>
	<img src=https://img-blog.csdnimg.cn/0625a901f2064232855327df8615bfef.png
	width=80%>
</div>


接下来将一辆正确分类的汽车图片（图 5）叠加到图 4 蓝色响应区域（即模型并不关注的区域），结果如图 6 所示，汽车部分的响应值很小，模型仍通过天蓝色区域将图片判为了飞机。接着又将汽车叠加到图 4 红色响应区域（图的右下角），结果如图 7 所示，仍将图片判为了飞机。
有意思的是将汽车叠加到图 7 的红色响应区域，模型把图片判为了船，而且红色响应区域是蓝色区域的下部分，这个与船在大海中的位置很接近。

通过以上代码学习 `backward_hook` 的使用及其在 Grad-CAM 中的应用，并通过 Grad-CAM 能诊断模型是否学习到了关键特征。

# 知识来源

1. [【05-05-hook函数与CAM算法.mp4】](https://www.bilibili.com/video/BV1uL4y1A76E?vd_source=ac73c03faf1b37a5bc2296969f45cf7b)
2. [PyTorch的hook及其在Grad-CAM中的应用_PyTorch 定义hook观察grad-CSDN博客](https://blog.csdn.net/u011995719/article/details/97752853)
