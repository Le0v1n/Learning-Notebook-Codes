# 1. 前言

注册机制是一种在编程中常见的设计模式，<u>它允许程序在运行时动态地将函数、类或其他对象注册到某个中心管理器中，以便随后可以使用这些注册的对象</u>。在Python中，注册机制通常用于**实现插件系统**、**扩展性架构**以及**回调函数的管理**。

通俗的说，当我们的项目中需要成批量的函数和类，且这些函数和类功能上相似或并行时，为了方便管理，我们可以把这些指定的函数和类整合到一个字典。我们可以用函数名或类名作为字典的 `key`，也可用使用自定义的名字作为 `key`，对应的函数或类作为 `value`。构建这样一个字典的过程就是注册（Registry），Python 引入注册器机制保证了这个字典可以**自动维护**，<font color='red'>增加或删除新的函数或类时，不需要手动去修改字典</font>。

Python 注册器机制本质上是用装饰器（decorator）来实现的。下面我们将从基本的 Python 函数出发，逐步介绍装饰器，最后来学习注册器。

# 1. 理解 Python 函数

## 1.1 函数的不同调用

首先定义一个函数，然后用不同的方式调用它。

```python
def foo():
    return "I am Le0v1n"

print(f"foo(): {foo()}")


fn = foo  # 这里 foo 后面没有小括号，不是函数调用，而是将 foo 函数赋值给变量 fn
print(f"fn(): {fn()}")
```

```
foo(): I am Le0v1n
fn(): I am Le0v1n
```

## 1.2 函数中的函数

在函数体中还可以定义函数（函数的函数:joy:），只是这个函数体内的函数不能在函数体外被直接调用：

```python
def foo():
    print("foo 函数正在运行...")
    
    # 定义函数中的函数
    def bar():
        return "foo.bar 函数正在运行..."
    
    def bam():
        return "foo.bam 函数正在运行..."
        
    # 调用函数中的函数
    print(bar())
    print(bam())
    print("foo 函数即将结束!")
    

if __name__ == "__main__":
    foo()
```

```
foo 函数正在运行...
foo.bar 函数正在运行...
foo.bam 函数正在运行...
foo 函数即将结束!
```

上面的结果没有什么意思，但如果我们直接调用函数的函数，会发生什么？

```python
# 如果我们调用函数中的函数
try:
    bar()
except Exception as e:
    print(f"报错啦: {e}")
    
try:
    bam()
except Exception as e:
    print(f"报错啦: {e}")
```

```
报错啦: name 'bar' is not defined
报错啦: name 'bam' is not defined
```

## 1.3 函数中函数的外部调用

函数体内的函数虽然不能在函数体外被直接调用，但是可以将它们返回出来。

```python
def foo(choice='bar'):
    print("foo 函数正在运行...")

    # 定义函数中的函数
    def bar():
        return "foo.bar 函数正在运行..."

    def bam():
        return "foo.bam 函数正在运行..."

    print("foo 函数即将结束!")

    if choice == 'bar':
        return bar
    elif choice == 'bam':
        return bam
    else:
        raise NotImplementedError("choice 必须是 bar 或 bam !")


if __name__ == "__main__":
    fn1 = foo(choice='bar')
    fn2 = foo(choice='bam')
    print(fn1)
    print(fn2)
    print(fn1())
    print(fn2())
```

```
foo 函数正在运行...
foo 函数即将结束!
foo 函数正在运行...
foo 函数即将结束!
<function foo.<locals>.bar at 0x000001D72F1ECE50>
<function foo.<locals>.bam at 0x000001D72F2558B0>
foo.bar 函数正在运行...
foo.bam 函数正在运行...
```

注意到返回的 `bar` 和 `bam` 后面没有小括号，那它就可以被传递，并且可以赋值给别的变量而被执行，如果有小括号，那它就会被执行。

## 1.4 函数作为函数参数

我们还可以将函数作为参数传递给另一个函数：

```python
def foo():
    return "I am foo"

def bar(fn):
    print("I am bar")
    print(fn())
    
    
if __name__ == "__main__":
    bar(foo)
    print()
    
    try:
        bar(foo())
    except Exception as e:
        print(f"报错啦: {e}")
```

```
I am bar
I am foo

I am bar
报错啦: 'str' object is not callable
```

## 1.5 函数的包装

有了这样的印象之后，我们再写一个更加复杂一点的例子：

```python
def decorator(fn):
    def wrapper():
        print("---------- 函数调用前 ----------")
        fn()  # 调用函数
        print("---------- 函数调用后 ----------")
    return wrapper


def foo():
    print("I am foo!")
    
    
if __name__ == "__main__":
    # 直接调用函数
    print("直接调用函数: ", end="")
    foo()
    print()

    # 调用装饰器包装后函数
    fn = decorator(foo)  # 将foo函数用装饰器包装 -> fn
    print("调用包装后的foo函数: ")
    fn()  # 调用包装后的foo函数
    print()
```

```
直接调用函数: I am foo!

调用包装后的foo函数: 
---------- 函数调用前 ----------
I am foo!
---------- 函数调用后 ----------
```

# 2. 理解 Python 装饰器

## 2.1 定义

Python 装饰器是一种高阶函数，用于修改其他函数的行为或添加额外功能。装饰器本质上是一个函数，它接受一个函数作为参数，然后返回一个新的函数，通常扩展了或修改了原始函数的行为。

## 2.2 装饰器的初步使用

上一节的最后一个例子我们封装了一个函数 `foo`，并且用另一个函数 `decorate` 去修改这个函数的行为，这个功能其实就是 Python 装饰器（Decorate）所做的事情，只是我们以函数的形式显式的写了出来。Python 中的装饰器提供了更简洁的方式来实现同样的功能，装饰器的写法是在被装饰的函数前使用 `@装饰器名`。现在我们用装饰器的写法来实现同样的功能：

```python
def decorator(fn):
    def wrapper():
        print("---------- 函数调用前 ----------")
        fn()
        print("---------- 函数调用后 ----------")

    return wrapper
    

@decorator  # @装饰器名称
def foo():
    print("I am foo")
    
    
if __name__ == "__main__":
    # 直接调用被装饰的函数
    print("直接调用函数: ")
    foo()
    print()

    print(f"函数的名称: {foo.__name__}")
```

```
直接调用函数: 
---------- 函数调用前 ----------
I am foo
---------- 函数调用后 ----------

函数的名称: wrapper
```

可以看到，当我们使用 `@装饰器名称` 对 `foo` 函数进行装饰后，直接调用 `foo` 函数就可以达到之前的效果，这无疑更加方便。

> **decorate**:
> + 英[`ˈdekəreɪt`] 美[`ˈdekəreɪt`]
> + v. 装饰; 装潢; 点缀; 装点; 油漆; 粉刷; 糊墙纸; 授给（某人）勋章（或奖章）;
>
> **wrapper**:
> + 英[`ˈræpə(r)`] 美[`ˈræpər`]
> + n. 封套; （食品等的）包装材料; 包装纸; 封皮;

---

**Q**: 为什么 `@decorate` 不能加 `( )`？
**A**: `@decorator` 不能加 `( )` 是因为 `@decorator` 是用来应用装饰器的语法糖，**而不是直接调用装饰器函数**。装饰器通常是函数，它接受一个函数作为参数，然后返回一个新的函数，用于包装原始函数。因此，在使用装饰器时，应该省略 `( )`。

当我们使用 `@decorator` 这种语法时，Python 实际上会将被装饰的函数（`foo`）作为参数传递给 `decorator` 函数。然后，`decorator` 函数返回一个包装了 `foo` 的新函数 `wrapper`，而不是直接调用 `decorator`。这允许我们在 `foo` 函数的前后添加额外的操作，而不需要显式地调用 `decorator`。

如果我们在 `@decorator` 后面加上 `()`，就变成了直接调用 `decorator` 函数，而不是应用装饰器。这通常不是我们的意图，因为我们的目标是装饰 `foo` 函数，而不是调用 `decorator`。正确的做法是使用 `@decorator` 而不带括号，如我们在提供的示例中所示。

## 2.3 装饰器的问题

与此同时，我们也发现了一个问题：当我们输出被装饰函数的名字时，它被 `wrapper` 函数替代了。如果我们需要获取调用函数的名称，此时输出 `wrapper` 是不合适的。Python 为了解决这个问题，提供了一个简单的函数 `functools.wraps`。

```python
from functools import wraps


def decorator(fn):
    @wraps(fn)
    def wrapper():
        print("---------- 函数调用前 ----------")
        fn()
        print("---------- 函数调用后 ----------")
    return wrapper


@decorator
def foo():
    print("I am foo")


if __name__ == "__main__":
    print("直接调用函数: ")
    foo()
    print()

    print(f"函数的名称: {foo.__name__}")
```

```
直接调用函数: 
---------- 函数调用前 ----------
I am foo
---------- 函数调用后 ----------

函数的名称: foo
```

---

**Q**：为什么 `@wraps` 要加 `(fn)`？
**A**：`@wraps` 装饰器用于保留原始函数的元信息，例如函数的名称 (`__name__`)、文档字符串 (`__doc__`) 等，以确保包装函数在行为上与原始函数一致，并且在使用工具或调试时提供准确的信息。<font color='red'>`@wraps` 需要传递原始函数 (`fn`) 作为参数，以便它知道应该保留哪个函数的元信息</font>。

在我们的示例中，`@wraps(fn)` 装饰了 `wrapper` 函数，其中 `fn` 是被装饰的原始函数，也就是 `foo`。这告诉 `@wraps` 装饰器将 `wrapper` 的元信息设置为与 `foo` 相关的元信息，以确保 `wrapper` 在元信息上与 `foo` 一致。

如果不使用 `@wraps` 装饰器，`wrapper` 函数将继承 `decorator` 函数的元信息，而不是 `foo` 函数的元信息，这可能导致在使用 `foo` 时出现不一致或不正确的元信息。所以，`@wraps(fn)` 用于解决这个问题，确保包装函数的正确元信息。

## 2.4 类的装饰器

不仅仅只有函数可以构建装饰器，类也可以用于构建装饰器，在构建装饰器类时，需要将原本装饰器函数的部分实现于 `__call__` 函数中即可：

```python
from functools import wraps


class Decorate:
    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self):
        @wraps(self.fn)
        def wrapper(*args, **kwargs):
            print("---------- 函数调用前 ----------")
            self.fn(*args, **kwargs)
            print("---------- 函数调用后 ----------")
        return wrapper


@Decorate  # 用类来装饰函数，那么函数也变为了类
def foo(param1, param2):
    print(f"I am foo. \n"
          f"My parameters are: \n"
          f"param1: {param1} | param2: {param2}")


if __name__ == "__main__":
    # 实例化类对象
    obj = foo()

    # 调用对象的方法
    obj("参数1", "参数2")
```

```
---------- 函数调用前 ----------
I am foo. 
My parameters are: 
param1: 参数1 | param2: 参数2
---------- 函数调用后 ----------
```

# 3. Python 注册器 —— Registry

## 3.1 实现一个手动注册器

有了装饰器的基础之后，我们现在要走入注册器的世界了。Python 的注册器本质上就是用装饰器的原理实现的。Registry 提供了字符串到函数或类的映射，这个映射会被整合到一个字典中，开发者只要输入输入相应的字符串（为函数或类起的名字）和参数，就能获得一个函数或初始化好的类。为了说明 Registry 的好处，我们首先看一下用一个字典存放字符串到函数的映射：

```python
def foo():
    ...


def fn(x): return x**2


class ExampleClass:
    ...


if __name__ == "__main__":
    # 创建注册字典
    register = dict()

    # 开始为函数和类进行注册
    register[foo.__name__] = foo
    register[fn.__name__] = fn
    register[ExampleClass.__name__] = ExampleClass

    print(register)
```

```
{'foo': <function foo at 0x000001D730752550>, 
'<lambda>': <function <lambda> at 0x000001D730752D30>, 
'ExmpleClass': <class '__main__.ExampleClass'>} 
```

虽然这样也可以创建一个注册器，但这样做的缺点是我们需要手动维护 `register` 这个字典，当增加或删除新的函数或类时，我们需要手动修改 `register` 这个字典，因此我们需要一个可以自动维护的字典，在我们定义一个函数或类的时候就自动把它整合到字典中。为了达到这一目的，这里就使用到了装饰器，<font color='blue'>在装饰器中将我们新定义的函数或类存放的字典中，这个过程我们称之为注册</font>。

## 3.2 实现一个半自动注册器

### 3.2.1 代码

这里我们需要定义一个装饰器类 `Register`，其中核心部分就是成员函数 `register`，它作为一个装饰器函数：

```python
class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = dict()  # 创建一个字典用于保存注册的可调用对象

    def register(self, target):
        def add_item(key, value):
            if key in self._dict:  # 如果 key 已经存在
                print(f"\033[31m"
                      f"WARNING: {value.__name__} 已经存在!"
                      f"\033[0m")

            # 进行注册，将 key 和 value 添加到字典中
            self[key] = value
            return value

        # 传入的 target 可调用 --> 没有给注册名 --> 传入的函数名或类名作为注册名
        if callable(target):  # key 为函数/类的名称; value 为函数/类本体
            return add_item(key=target.__name__, value=target)
        else:  # 传入的 target 不可调用 --> 抛出异常
            raise TypeError("\033[31mOnly support callable object, e.g. function or class\033[0m")
        
    def __call__(self, target):
        return self.register(target)

    def __setitem__(self, key, value):  # 将键值对添加到 _dict 字典中
        self._dict[key] = value

    def __getitem__(self, key):  # 从 _dict 字典中获取注册的可调用对象
        return self._dict[key]

    def __contains__(self, key):  # 检查给定的注册名是否存在于 _dict 字典中
        return key in self._dict

    def __str__(self):  # 返回 _dict 字典的字符串表示
        return str(self._dict)

    def keys(self):  # 返回 _dict 字典中的所有键
        return self._dict.keys()

    def values(self):  # 返回 _dict 字典中的所有值
        return self._dict.values()

    def items(self):  # 返回 _dict 字典中的所有键值对
        return self._dict.items()


if __name__ == "__main__":
    register_obj = Register()
    
    @register_obj  # 不用再 register_obj.register 了
    def fn1_add(a, b):
        return a + b
    
    @register_obj  # 不用再 register_obj.register 了
    def fn2_subject(a, b):
        return a - b
    
    @register_obj  # 不用再 register_obj.register 了
    def fn3_multiply(a, b):
        return a * b
    
    @register_obj  # 不用再 register_obj.register 了
    def fn4_divide(a, b):
        return a / b
    
    # 我们再重复定义一个函数
    @register_obj  # 不用再 register_obj.register 了
    def fn2_subject(a, b):
        return b - a
    
    # 尝试使用 register 方法注册不可调用的对象
    try:
        register_obj.register("传入字符串，它是不可调用的")
    except Exception as e:
        print(f"报错啦: {e}")

    print("所有函数均已注册!\n")
    
    # 我们查看一个注册器中有哪些元素
    print(f"\033[34mkey\t\tvalue\033[0m")
    for k, v in register_obj.items():  # <=> for k, v in register_obj._dict.items()
        print(f"{k}: \t{v}")
```

![](./imgs_markdown/2023-10-17-14-15-43.png)


## 3.3 实现一个全自动注册器

### 3.3.1 代码实现

如果不想手动调用 `register()` 函数，可以在 `Register` 类中添加一个 `__call__()` 函数：

```python
class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = dict()  # 创建一个字典用于保存注册的可调用对象

    def register(self, target):
        def add_item(key, value):
            if key in self._dict:  # 如果 key 已经存在
                print(f"\033[34m"
                      f"WARNING: {value.__name__} 已经存在!"
                      f"\033[0m")

            # 进行注册，将 key 和 value 添加到字典中
            self[key] = value
            return value

        # 传入的 target 可调用 --> 没有给注册名 --> 传入的函数名或类名作为注册名
        if callable(target):  # key 为函数/类的名称; value 为函数/类本体
            return add_item(key=target.__name__, value=target)
        else:  # 传入的 target 不可调用 --> 抛出异常
            raise TypeError("\033[31mOnly support callable object, e.g. function or class\033[0m")
        
    def __call__(self, target):
        return self.register(target)

    def __setitem__(self, key, value):  # 将键值对添加到 _dict 字典中
        self._dict[key] = value

    def __getitem__(self, key):  # 从 _dict 字典中获取注册的可调用对象
        return self._dict[key]

    def __contains__(self, key):  # 检查给定的注册名是否存在于 _dict 字典中
        return key in self._dict

    def __str__(self):  # 返回 _dict 字典的字符串表示
        return str(self._dict)

    def keys(self):  # 返回 _dict 字典中的所有键
        return self._dict.keys()

    def values(self):  # 返回 _dict 字典中的所有值
        return self._dict.values()

    def items(self):  # 返回 _dict 字典中的所有键值对
        return self._dict.items()


if __name__ == "__main__":
    register_obj = Register()
    
    @register_obj  # 不用再 register_obj.register 了
    def fn1_add(a, b):
        return a + b
    
    @register_obj  # 不用再 register_obj.register 了
    def fn2_subject(a, b):
        return a - b
    
    @register_obj  # 不用再 register_obj.register 了
    def fn3_multiply(a, b):
        return a * b
    
    @register_obj  # 不用再 register_obj.register 了
    def fn4_divide(a, b):
        return a / b
    
    # 我们再重复定义一个函数
    @register_obj  # 不用再 register_obj.register 了
    def fn2_subject(a, b):
        return b - a
    
    # 尝试使用 register 方法注册不可调用的对象
    try:
        register_obj("传入字符串，它是不可调用的")
        # <=> register_obj.register("传入字符串，它是不可调用的")  # 因为我们实现了__call__()函数
    except Exception as e:
        print(f"报错啦: {e}")

    print("\n所有函数均已注册!\n")
    
    # 我们查看一个注册器中有哪些元素
    print(f"\033[34mkey\t\tvalue\033[0m")
    for k, v in register_obj.items():  # <=> for k, v in register_obj._dict.items()
        print(f"{k}: \t{v}")
```

![](./imgs_markdown/2023-10-17-14-15-43.png)

### 3.3.2 代码分析

![](./Python的Registry.png)

> 高清图片链接: [图片 + 源文件](https://github.com/Le0v1n/Learning-Notebook-Codes/blob/main/Python_Registry/Python%E7%9A%84Registry.png)

1. `Register` 类继承了内置的 `dict` 类，并在其构造函数中初始化一个名为 `_dict` 的字典，用于保存注册的可调用对象。

2. `register` 方法用于注册可调用对象。它接受一个参数 `target`，这可以是可调用对象或者是一个注册名。如果 `target` 是可调用对象，它会将函数或类名作为注册名。如果 `target` 不可调用，它会将传入的注册名与传入的可调用对象关联。

3. `add_item` 内部函数检查可调用对象是否可被调用，如果不可调用会引发异常。它还检查注册名是否已存在，如果存在则发出警告。

4. :star: `__call__` 方法允许对象实例（`register_obj`）像函数一样被调用，实际上是将调用委托给 `register` 方法。

5. 其余的魔法方法（`__setitem__`, `__getitem__`, `__contains__`, `__str__`, `keys()`, `values()`, `items()`）覆盖了字典的行为，以便访问和管理内部的 `_dict` 字典。

6. 在主程序中，`Register` 类的一个实例 `register_obj` 被创建。

7. 使用装饰器 `@register_obj`，将多个函数注册到 `register_obj` 实例中，每个函数都有一个注册名。如果函数的注册名已经存在，会打印警告信息。

8. 最后，程序输出注册器中的所有注册名和可调用对象。

# 4. Python 注册器在深度学习中的应用

## 4.1 应用场景

在深度学习和机器学习中，注册器模式可以有一些有趣的应用，尤其是在构建自定义层、损失函数、优化器或其他模型组件时。以下是在深度学习中使用注册器的一些潜在应用示例：

1. 自定义层和模型
2. 自定义损失函数
3. 自定义优化器
4. 数据预处理步骤
5. 回调函数

这些示例说明了如何使用注册器模式来管理和选择深度学习中的各种组件，从而使模型的构建和训练更加灵活和可配置。通过注册器，我们可以轻松地扩展和定制深度学习模型的各个部分。

## 4.2 自定义层和模型

我们可以使用注册器来注册自定义神经网络层或模型结构。这在构建自定义神经网络架构时非常有用。例如，我们可以构建一个注册器，用于注册各种自定义层，如卷积层、循环层等。然后，我们可以在模型构建过程中按名称选择并使用这些自定义层。

```python
import torch.nn as nn

# 实现一个注册器
class LayerRegistry:
    def __init__(self):
        self.layers = dict()

    def register(self, layer_name):
        # 让装饰器接受 layer 参数
        def decorator(layer):
            # 开始注册
            self.layers[layer_name] = layer
            return layer  # 返回注册的层
        return decorator

    def get_layer(self, layer_name):
        if layer_name in self.layers:
            return self.layers[layer_name]
        else:
            raise KeyError(f"未注册的层 '{layer_name}'.")

# 实例化自定义层注册器
layer_register = LayerRegistry()

# 自定义层类
@layer_register.register("ConvBNReLU")
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    # 在创建层的使用可以使用注册器中的层
    example_layer = layer_register.get_layer("ConvBNReLU")
    
    # 创建具体的层实例
    specific_layer = example_layer(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
    
    # 打印具体层的信息
    print(specific_layer)
```

```
ConvBNReLU(
  (layers): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
)
```

---

但是这样似乎没有什么意思，是的，这样的确没有什么意义，一般注册器和配置文件一起用的，下面我们看一下例子：

```python
import torch.nn as nn


class LayerRegistry:  # 实现一个注册器
    def __init__(self):
        self.layers = dict()

    def register(self, layer_name):
        # 让装饰器接受 layer 参数
        def decorator(layer):
            # 开始注册
            self.layers[layer_name] = layer
            return layer  # 返回注册的层
        return decorator

    def get_layer(self, layer_name):
        if layer_name in self.layers:
            return self.layers[layer_name]
        else:
            raise KeyError(f"未注册的层 '{layer_name}'.")


# 实例化自定义层注册器
layer_register = LayerRegistry()


@layer_register.register("ConvBNReLU")
class ConvBNReLU(nn.Module):  # 自定义层类
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBNReLU, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


# 继续注册其他模块
@layer_register.register("BatchNorm2d")
class BatchNorm2d(nn.Module):
    def __init__(self, num_features, *args, **kwargs):
        super(BatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, *args, **kwargs)

    def forward(self, x):
        return self.bn(x)


@layer_register.register("ReLU")
class ReLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ReLU, self).__init__()
        self.relu = nn.ReLU(*args, **kwargs)

    def forward(self, x):
        return self.relu(x)


@layer_register.register("MaxPooling")
class MaxPooling(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(MaxPooling, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.maxpool(x)


@layer_register.register("AvgPooling")
class AvgPooling(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(AvgPooling, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.avgpool(x)


# 定义网络配置(cfg)来构建完整的网络
cfg = [
    ('ConvBNReLU', 3, 64, 3, 1),  # 传递4个参数
    ('MaxPooling', 2, 2, 0),
    ('ConvBNReLU', 64, 128, 3, 1),
    ('MaxPooling', 2, 2, 0),
    ('ConvBNReLU', 128, 256, 3, 1),
    ('AvgPooling', 4, 1, 0),
]


# 构建网络
class CustomNet(nn.Module):
    def __init__(self, cfg):
        super(CustomNet, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = 3  # 输入通道数

        for layer_cfg in cfg:
            layer_name, *layer_params = layer_cfg
            layer = layer_register.get_layer(layer_name)

            if layer_name in ['ConvBNReLU', 'BatchNorm2d']:
                self.layers.append(layer(in_channels, *layer_params))
                in_channels = layer_params[1]
            else:
                self.layers.append(layer(*layer_params))



# 创建完整的网络实例
custom_net = CustomNet(cfg)

# 打印网络结构
print(custom_net)
```

```
CustomNet(
  (layers): ModuleList(
    (0): ConvBNReLU(
      (layers): Sequential(
        (0): Conv2d(3, 3, kernel_size=(64, 64), stride=(3, 3), padding=(1, 1))
        (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
    (1): MaxPooling(
      (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (2): ConvBNReLU(
      (layers): Sequential(
        (0): Conv2d(64, 64, kernel_size=(128, 128), stride=(3, 3), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
    (3): MaxPooling(
      (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (4): ConvBNReLU(
      (layers): Sequential(
        (0): Conv2d(128, 128, kernel_size=(256, 256), stride=(3, 3), padding=(1, 1))
...
      (avgpool): AvgPool2d(kernel_size=4, stride=1, padding=0)
    )
  )
)
```

我们可以看到，模型根据 `cfg` 变量搭建了模型，那么我们就可以通过读取 `.cfg/.config/.yaml/.yml/.json` 等格式的配置文件，从而非常方便的搭建模型或者修改模型。

## 4.3 自定义损失函数

深度学习任务通常需要特定的损失函数。通过使用注册器，我们可以注册和管理各种自定义损失函数，并在模型编译时选择要使用的损失函数。

我们可以仿照上面的例子，使用配置文件的方式注册损失函数。首先，我们需要创建一个类似的注册器类，然后在配置文件中定义不同的损失函数以及它们的参数。下面是一个示例：

首先，创建一个注册器类 `LossRegistry`：

```python
class LossRegistry:
    def __init__(self):
        self.losses = dict()

    def register(self, loss_name):
        def decorator(loss_fn):
            self.losses[loss_name] = loss_fn
            return loss_fn
        return decorator

    def get_loss(self, loss_name):
        if loss_name in self.losses:
            return self.losses[loss_name]
        else:
            raise KeyError(f"未注册的损失函数 '{loss_name}'.")

# 实例化自定义损失函数注册器
loss_register = LossRegistry()
```

然后，我们可以创建不同的损失函数并注册它们：

```python
@loss_register.register("MSE")
class MeanSquaredErrorLoss(nn.Module):
    def forward(self, input, target):
        return nn.functional.mse_loss(input, target)

@loss_register.register("CE")
class CrossEntropyLoss(nn.Module):
    def forward(self, input, target):
        return nn.functional.cross_entropy(input, target)
```

接下来，我们可以使用配置文件定义不同的损失函数：

```python
loss_config = [
    ('MSE', None),  # 使用默认参数
    ('CE', None),   # 使用默认参数
]

# 通过配置文件构建损失函数列表
loss_functions = [loss_register.get_loss(loss_name) for loss_name, _ in loss_config]

loss_fn_1 = loss_functions[0]()
loss_fn_2 = loss_functions[1]()
print(loss_fn_1)
print(loss_fn_2)
```

```
MeanSquaredErrorLoss()
CrossEntropyLoss()
```

现在，我们可以使用 `loss_functions` 列表中的损失函数来定义我们的损失函数组合。这样，我们可以根据配置文件轻松切换不同的损失函数，而无需更改网络代码。


## 4.4 自定义优化器

与损失函数一样，我们可以注册自定义优化器并在模型编译时选择要使用的优化器。这可以让我们尝试不同的优化算法，并根据任务选择最合适的优化器。

我们可以使用与损失函数注册类似的方法来注册不同的优化器。首先，创建一个注册器类 `OptimizerRegistry`，然后在配置文件中定义不同的优化器及其参数。以下是一个示例：

首先，创建一个注册器类 `OptimizerRegistry`：

```python
import torch.optim as optim

class OptimizerRegistry:
    def __init__(self):
        self.optimizers = dict()

    def register(self, optimizer_name):
        def decorator(optimizer_fn):
            self.optimizers[optimizer_name] = optimizer_fn
            return optimizer_fn
        return decorator

    def get_optimizer(self, optimizer_name, model_parameters, *args, **kwargs):
        if optimizer_name in self.optimizers:
            return self.optimizers[optimizer_name](model_parameters, *args, **kwargs)
        else:
            raise KeyError(f"未注册的优化器 '{optimizer_name}'.")

# 实例化自定义优化器注册器
optimizer_register = OptimizerRegistry()
```

然后，我们可以创建不同的优化器并注册它们：

```python
@optimizer_register.register("SGD")
class SGDOptimizer:
    def __init__(self, model_parameters, lr, momentum):
        self.optimizer = optim.SGD(model_parameters, lr=lr, momentum=momentum)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

@optimizer_register.register("Adam")
class AdamOptimizer:
    def __init__(self, model_parameters, lr, betas):
        self.optimizer = optim.Adam(model_parameters, lr=lr, betas=betas)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
```

接下来，我们可以使用配置文件定义不同的优化器：

```python
optimizer_config = [
    ('SGD', {'lr': 0.01, 'momentum': 0.9}),
    ('Adam', {'lr': 0.001, 'betas': (0.9, 0.999)})
]

# 通过配置文件构建优化器列表
optimizers = [optimizer_register.get_optimizer(optimizer_name, model_parameters, **params) for optimizer_name, params in optimizer_config]

for optimizer in optimizers:
    optimizer.zero_grad()  # 清空梯度
    optimizer.step()  # 下一步
```

现在，我们可以使用 `optimizers` 列表中的不同优化器来为模型定义不同的优化器。这使我们可以根据配置文件轻松切换不同的优化器，而无需更改网络代码。

## 4.5 数据预处理步骤

在深度学习中，数据预处理对于模型性能非常重要。我们可以注册各种数据预处理步骤，例如图像增强、标准化方法等，然后根据需要应用它们。

数据预处理是深度学习中的重要步骤之一。我们可以使用与前面示例类似的方法来注册不同的数据预处理步骤。首先，创建一个注册器类 `PreprocessingRegistry`，然后在配置文件中定义不同的数据预处理步骤及其参数。以下是一个示例：

首先，创建一个注册器类 `PreprocessingRegistry`：

```python
class PreprocessingRegistry:
    def __init__(self):
        self.preprocessing_steps = dict()

    def register(self, step_name):
        def decorator(preprocessing_fn):
            self.preprocessing_steps[step_name] = preprocessing_fn
            return preprocessing_fn
        return decorator

    def get_preprocessing_step(self, step_name, *args, **kwargs):
        if step_name in self.preprocessing_steps:
            return self.preprocessing_steps[step_name](*args, **kwargs)
        else:
            raise KeyError(f"未注册的数据预处理步骤 '{step_name}'.")

# 实例化自定义数据预处理步骤注册器
preprocessing_register = PreprocessingRegistry()
```

然后，我们可以创建不同的数据预处理步骤并注册它们：

```python
import numpy as np

@preprocessing_register.register("Normalize")
class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        return (data - self.mean) / self.std

@preprocessing_register.register("RandomCrop")
class RandomCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, data):
        h, w, c = data.shape
        x = np.random.randint(0, h - self.crop_size)
        y = np.random.randint(0, w - self.crop_size)
        return data[x:x+self.crop_size, y:y+self.crop_size, :]

@preprocessing_register.register("Resize")
class Resize:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, data):
        return cv2.resize(data, (self.target_size, self.target_size))
```

接下来，我们可以使用配置文件定义不同的数据预处理步骤：

```python
preprocessing_config = [
    ('Normalize', {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}),
    ('RandomCrop', {'crop_size': 224}),
    ('Resize', {'target_size': 256})
]

# 通过配置文件构建数据预处理步骤列表
preprocessing_steps = [preprocessing_register.get_preprocessing_step(step_name, **params) for step_name, params in preprocessing_config]
```

现在，我们可以使用 `preprocessing_steps` 列表中的不同数据预处理步骤来预处理我们的数据。这使我们可以根据配置文件轻松切换不同的数据预处理步骤，而无需更改数据处理代码。

```python
# 假设我们有一张原始图像
original_image = cv2.imread('./lena.png')  # 读取原始图像

# 应用数据预处理步骤
preprocessed_data = original_image.copy()  # 创建副本以保存经过预处理的数据

for preprocessing_step in preprocessing_steps:
    preprocessed_data = preprocessing_step(preprocessed_data)

# preprocessed_data 现在包含了经过预处理的数据
print(preprocessed_data.shape)  # (256, 256, 3)

# 现在可以将 preprocessed_data 用于深度学习模型的训练或推理
```


## 4.6 回调函数

在深度学习训练中，回调函数用于执行各种操作，如保存模型检查点、记录训练指标、可视化等。我们可以使用注册器来注册各种回调函数，并在模型训练时选择适当的回调函数。


首先，创建一个回调函数注册器类 `CallbackRegistry`：

```python
class CallbackRegistry:
    def __init__(self):
        self.callbacks = dict()

    def register(self, callback_name):
        def decorator(callback_fn):
            self.callbacks[callback_name] = callback_fn
            return callback_fn
        return decorator

    def get_callback(self, callback_name):
        if callback_name in self.callbacks:
            return self.callbacks[callback_name]
        else:
            raise KeyError(f"未注册的回调函数 '{callback_name}'.")

# 实例化自定义回调函数注册器
callback_register = CallbackRegistry()
```

接下来，我们可以创建不同的回调函数并注册它们，例如，一个用于保存模型检查点的回调函数和一个用于记录训练指标的回调函数：

```python
import os

@callback_register.register("ModelCheckpoint")
class ModelCheckpointCallback:
    def __init__(self, save_dir, save_freq):
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.best_accuracy = 0.0
        self.model = None

    def on_epoch_end(self, model, epoch, val_accuracy):
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.model = model
            model_save_path = os.path.join(self.save_dir, f"best_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_save_path)

@callback_register.register("RecordMetrics")
class RecordMetricsCallback:
    def __init__(self, log_file):
        self.log_file = log_file

    def on_epoch_end(self, epoch, train_loss, val_loss, train_accuracy, val_accuracy):
        with open(self.log_file, 'a') as file:
            file.write(f"Epoch {epoch} - Train Loss: {train_loss}, Val Loss: {val_loss}, Train Accuracy: {train_accuracy}, Val Accuracy: {val_accuracy}\n")
```

现在，我们可以在模型训练时选择适当的回调函数，并在适当的时机调用它们：

```python
# 在训练循环中选择适当的回调函数
for epoch in range(num_epochs):
    train_loss, train_accuracy = train_one_epoch()
    val_loss, val_accuracy = validate()

    # 在每个 epoch 结束时调用回调函数
    for callback_name, callback_instance in callback_instances.items():
        callback_instance.on_epoch_end(epoch, train_loss, val_loss, train_accuracy, val_accuracy)
```

上述示例中，我们注册了两种回调函数，一个用于保存模型检查点，另一个用于记录训练指标。然后，在训练循环中，在每个 epoch 结束时调用这些回调函数，以执行相应的操作。

我们可以根据需要定义更多的回调函数，并根据模型训练的具体需求来选择和调用它们。

# 知识来源
1. [【Python】Python的Registry机制](https://zhuanlan.zhihu.com/p/567619814)