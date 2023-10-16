当你的项目中需要成批量的函数和类，且这些函数和类功能上相似或并行时，为了方便管理，你可以把这些指定的函数和类整合到一个字典，你可以用函数名或类名作为字典的 `key`，也可用使用自定义的名字作为 `key`，对应的函数或类作为 `value`。构建这样一个字典的过程就是注册（Registry），Python 引入注册器机制保证了这个字典可以**自动维护**，<font color='red'>增加或删除新的函数或类时，不需要手动去修改字典</font>。

Python 注册器机制本质上是用装饰器（decorator）来实现的。下面我们将从基本的 Python 函数出发，逐步介绍装饰器，最后来学习注册器。

# 一、理解 Python 函数

首先定义一个函数，然后用不同的方式调用它。

```python
def foo():
    return "I am Le0v1n"

print(foo())  # I am Le0v1n


fn = foo
print(fn())  # I am Le0v1n
```

在函数体中还可以定义函数，只是这个函数体内的函数不能在函数体外被直接调用：














# 知识来源
1. [【Python】Python的Registry机制](https://zhuanlan.zhihu.com/p/567619814)