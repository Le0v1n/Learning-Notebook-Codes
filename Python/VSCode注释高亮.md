
| 注释格式   | 说明                                                                           |
| ---------- | ------------------------------------------------------------------------------ |
| `# NOTE:`  | 用于标记需要注意的地方，如重要信息、提示等                                     |
| `# TODO:`  | 用于标记待办事项，提醒开发者需要完成的任务                                     |
| `# FIXME:` | 用于标记需要修复的问题，通常是在代码中已知的错误或需要改进的地方               |
| `# XXX:`   | 用于标记代码中需要注意的地方，可能是一些潜在的问题或需要注意的细节             |
| `# HACK:`  | 用于标记一些临时的、不太优雅的解决方案，通常是为了快速解决问题而采取的权宜之计 |
| `# BUG:`   | 用于标记已知的bug，提醒开发者需要修复                                          |

```python
# NOTE: 这里是一个重要的配置项
important_config = True

# TODO: 添加异常处理逻辑
def divide(a, b):
    return a / b

# FIXME: 这里的循环逻辑有误，需要重新调整
def process_data(data):
    for i in range(len(data) + 1):  # 这里可能会越界
        print(data[i])

# XXX: 这里的参数值可能需要根据实际情况进行调整
def calculate_score(points, multiplier=2):
    return points * multiplier

# HACK: 暂时使用这种方法来绕过这个问题，后续需要找到更合适的解决方案
def quick_fix():
    # 这里是一个临时的解决方案
    pass

# BUG: 这里存在一个内存泄漏的问题
def memory_leak():
    # 这里是可能导致内存泄漏的代码
    pass
```

<center class="half">
    <img src=./imgs_markdown/VSCode注释高亮_light.png width=49%/>
    <img src=./imgs_markdown/VSCode注释高亮_dark.png width=49%/>
</center>
