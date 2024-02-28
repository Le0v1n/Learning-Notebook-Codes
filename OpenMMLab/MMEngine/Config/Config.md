# 1. Config 介绍

MMEngine 实现了抽象的配置类（Config），为用户提供统一的配置访问接口。配置类能够支持不同格式的配置文件，包括 python，json，yaml，用户可以根据需求选择自己偏好的格式。配置类提供了类似字典或者 Python 对象属性的访问接口，用户可以十分自然地进行配置字段的读取和修改。为了方便算法框架管理配置文件，配置类也实现了一些特性，例如配置文件的字段继承等。

# 2. 下载示例配置文件

在开始教程之前，我们先将教程中需要用到的配置文件下载到本地（建议在临时目录下执行，方便后续删除示例配置文件）：

```bash
wget -P OpenMMLab/MMEngine/Config/codes/example_configs https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/config_sgd.py
wget -P OpenMMLab/MMEngine/Config/codes/example_configs https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/cross_repo.py
wget -P OpenMMLab/MMEngine/Config/codes/example_configs https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/custom_imports.py
wget -P OpenMMLab/MMEngine/Config/codes/example_configs https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/demo_train.py
wget -P OpenMMLab/MMEngine/Config/codes/example_configs https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/example.py
wget -P OpenMMLab/MMEngine/Config/codes/example_configs https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/learn_read_config.py
wget -P OpenMMLab/MMEngine/Config/codes/example_configs https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/my_module.py
wget -P OpenMMLab/MMEngine/Config/codes/example_configs https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/optimizer_cfg.py
wget -P OpenMMLab/MMEngine/Config/codes/example_configs https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/predefined_var.py
wget -P OpenMMLab/MMEngine/Config/codes/example_configs https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/refer_base_var.py
wget -P OpenMMLab/MMEngine/Config/codes/example_configs https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/replace_data_root.py
wget -P OpenMMLab/MMEngine/Config/codes/example_configs https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/replace_num_classes.py
wget -P OpenMMLab/MMEngine/Config/codes/example_configs https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/resnet50_delete_key.py
wget -P OpenMMLab/MMEngine/Config/codes/example_configs https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/resnet50_lr0.01.py
wget -P OpenMMLab/MMEngine/Config/codes/example_configs https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/resnet50_runtime.py
wget -P OpenMMLab/MMEngine/Config/codes/example_configs https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/resnet50.py
wget -P OpenMMLab/MMEngine/Config/codes/example_configs https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/runtime_cfg.py
wget -P OpenMMLab/MMEngine/Config/codes/example_configs https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/modify_base_var.py
```

> 配置类支持两种风格的配置文件，即纯文本风格的配置文件和纯 Python 风格的配置文件（v0.8.0 的新特性），二者在调用接口统一的前提下各有特色。在一些情况下，纯文本风格的配置文件写法更加简洁，语法兼容性更好（json、yaml 通用）。如果希望配置文件的写法可以更加灵活，建议阅读并使用纯 Python 风格的配置文件（beta）

# 3. 配置文件读取

配置类提供了统一的接口 `Config.fromfile()`，来读取和解析配置文件。

合法的配置文件应该定义一系列<font color='red'><b>键值对</b></font>，这里举几个不同格式配置文件的例子。

Python 格式：

```python
test_int = 1
test_list = [1, 2, 3]
test_dict = dict(key1='value1', key2=0.1)
```

Json 格式：

```json
{
  "test_int": 1,
  "test_list": [1, 2, 3],
  "test_dict": {"key1": "value1", "key2": 0.1}
}
```

YAML 格式：

```yaml
test_int: 1
test_list: [1, 2, 3]
test_dict:
  key1: "value1"
  key2: 0.1
```

对于以上三种格式的文件，假设文件名分别为 `config.py`，`config.json`，`config.yml`，调用 `Config.fromfile('config.xxx')` 接口加载这三个文件都会得到相同的结果，构造了包含 3 个字段的配置对象。我们以 `config.py` 为例，我们先将示例配置文件下载到本地，然后通过配置类的 `fromfile` 接口读取配置文件：

```python
from mmengine.config import Config

cfg = Config.fromfile('OpenMMLab/MMEngine/codes/example_configs/learn_read_config.py')
print(cfg)
```

```
Config (path: OpenMMLab/MMEngine/codes/example_configs/learn_read_config.py): {'test_int': 1, 'test_list': [1, 2, 3], 'test_dict': {'key1': 'value1', 'key2': 0.1}}
```

# 4. 配置文件的使用

通过读取配置文件来初始化配置对象后，就**可以像使用普通字典或者 Python 类一样来使用这个变量了**。我们提供了两种访问接口，即类似字典的接口 `cfg['key']` 或者类似 Python 对象属性的接口 `cfg.key`。这两种接口都支持读写。

```python
import os
import sys
from mmengine.config import Config

sys.path.append(os.getcwd())
from utils.common_fn import xprint


cfg = Config.fromfile('OpenMMLab/MMEngine/codes/example_configs/learn_read_config.py')
xprint(cfg, color='red', hl='>')

xprint(cfg.test_int, color='green', hl='-', hl_style='full')
xprint(cfg['test_list'], color='green')
xprint(cfg['test_dict'], color='green')

cfg['test_list'][1] = 3  # 修改字典中的值
xprint(cfg['test_list'], color='green', hl='-', hl_style='full')
```

```
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Config (path: OpenMMLab/MMEngine/codes/example_configs/learn_read_config.py): {'test_int': 1, 'test_list': [1, 2, 3], 'test_dict': {'key1': 'value1', 'key2': 0.1}}
----------------------------------------------------------------------------------------------------------------------------------------------------
1
[1, 2, 3]
{'key1': 'value1', 'key2': 0.1}
----------------------------------------------------------------------------------------------------------------------------------------------------
[1, 3, 3]
```

💡 注意：配置文件中定义的嵌套字段（即类似字典的字段），在 `Config` 中会将其转化为 `ConfigDict` 类，该类继承了 Python 内置字典类型的全部接口，同时也支持以对象属性的方式访问数据。

在算法库中，可以将配置与注册器结合起来使用，达到通过配置文件来控制模块构造的目的。这里举一个在配置文件中定义优化器的例子。

假设我们已经定义了一个优化器的注册器 `OPTIMIZERS`，包括了各种优化器。那么首先写一个 `config_sgd.py`：

```python

```

