from pathlib import Path
import inspect
from prettytable import PrettyTable


path = 'Datasets/coco128'
p = Path(path)

# 查看类的方法和属性
ptab = PrettyTable(['性质', '用法', '说明'])
for name, member in inspect.getmembers(Path):
    if inspect.isfunction(member) or inspect.ismethod(member):
        if name.startswith(('_', '__')):
            continue
        else:
            ptab.add_row(['🧊 方法', '.'+name+'()', ''])
    elif not name.startswith("__"):
        if name.startswith(('_', '__')):
            continue
        else:
            ptab.add_row(['🛠️ 属性', '.'+name, ''])
print(ptab)
