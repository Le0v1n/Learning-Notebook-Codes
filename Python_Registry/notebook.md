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

1. `Register` 类继承了内置的 `dict` 类，并在其构造函数中初始化一个名为 `_dict` 的字典，用于保存注册的可调用对象。

2. `register` 方法用于注册可调用对象。它接受一个参数 `target`，这可以是可调用对象或者是一个注册名。如果 `target` 是可调用对象，它会将函数或类名作为注册名。如果 `target` 不可调用，它会将传入的注册名与传入的可调用对象关联。

3. `add_item` 内部函数检查可调用对象是否可被调用，如果不可调用会引发异常。它还检查注册名是否已存在，如果存在则发出警告。

4. :star: `__call__` 方法允许对象实例（`register_obj`）像函数一样被调用，实际上是将调用委托给 `register` 方法。

5. 其余的魔法方法（`__setitem__`, `__getitem__`, `__contains__`, `__str__`, `keys()`, `values()`, `items()`）覆盖了字典的行为，以便访问和管理内部的 `_dict` 字典。

6. 在主程序中，`Register` 类的一个实例 `register_obj` 被创建。

7. 使用装饰器 `@register_obj`，将多个函数注册到 `register_obj` 实例中，每个函数都有一个注册名。如果函数的注册名已经存在，会打印警告信息。

8. 最后，程序输出注册器中的所有注册名和可调用对象。

# 知识来源
1. [【Python】Python的Registry机制](https://zhuanlan.zhihu.com/p/567619814)