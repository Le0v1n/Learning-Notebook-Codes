from pathlib import Path
from prettytable import PrettyTable, MSWORD_FRIENDLY


path = 'Datasets/coco128'
p = Path(path)

ptab = PrettyTable(['性质', '用法', '结果', '数据类型', '说明'])
ptab.set_style(MSWORD_FRIENDLY)
ptab.align['用法'] = 'l'
ptab.align['结果'] = 'l'
ptab.align['数据类型'] = 'l'
ptab.align['说明'] = 'l'

# ------------------------------ 属性 ------------------------------
flag = '🛠️  属性'
ptab.add_row(['', 'p', p, type(p), 'Path的实例化对象'])
ptab.add_row([flag, 'p.anchor', p.anchor, type(p.anchor), '路径的“锚”，通常是驱动器或UNC共享'])
ptab.add_row([flag, 'p.drive', p.drive, type(p.drive), '返回路径的驱动器字母（如果有）'])
ptab.add_row([flag, 'p.name', p.name, type(p.name), '返回路径的最后一部分'])
ptab.add_row([flag, 'p.parent', p.parent, type(p.parent), '返回路径的父级目录（💡 还是一个Path对象）'])
ptab.add_row([flag, 'p.parts', p.parts, type(p.parts), '返回路径的组成部分'])
# ------------------------------------------------------------------

# ------------------------------ 方法 ------------------------------

# ------------------------------------------------------------------
print(ptab)