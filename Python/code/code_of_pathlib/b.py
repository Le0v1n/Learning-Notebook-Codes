from pathlib import Path
from prettytable import PrettyTable, MSWORD_FRIENDLY


filepath = '/mnt/d/Learning-Notebook-Codes/Datasets/coco128.tar.gz'
p = Path(filepath)
d = Path('Datasets/coco128/val')
f = Path('Datasets/coco128/train/labels/000000000572.txt')
base_path = Path('Datasets/coco128')

ptab = PrettyTable(['性质', '用法', '结果', '数据类型', '说明'])
ptab.set_style(MSWORD_FRIENDLY)
ptab.align['用法'] = 'l'
ptab.align['结果'] = 'l'
ptab.align['数据类型'] = 'l'
ptab.align['说明'] = 'l'

# ------------------------------ 属性 ------------------------------
flag = '🛠️  属性'
ptab.add_row(['', 'p', p, type(p), 'Path的实例化对象'])
ptab.add_row(['' for _ in range(5)])
ptab.add_row([flag, 'p.anchor', p.anchor, type(p.anchor), '路径的“锚”，通常是驱动器或UNC共享'])
ptab.add_row([flag, 'p.drive', p.drive, type(p.drive), '返回路径的驱动器字母（如果有）'])
ptab.add_row([flag, 'p.name', p.name, type(p.name), '返回路径的最后一部分'])
ptab.add_row([flag, 'p.parent', p.parent, type(p.parent), '返回路径的父级目录（💡 还是一个Path对象）'])
ptab.add_row([flag, 'p.parts', p.parts, type(p.parts), '返回路径的组成部分'])
ptab.add_row([flag, 'p.root', p.root, type(p.root), '返回路径的根部分（💡 如果是相对路径则为""）：'])
ptab.add_row([flag, 'p.stem', p.stem, type(p.stem), '返回没有后缀的文件名部分'])
ptab.add_row([flag, 'p.suffix', p.suffix, type(p.suffix), '返回文件扩展名'])
ptab.add_row([flag, 'p.suffixes', p.suffixes, type(p.suffixes), '返回文件所有后缀的列表'])
ptab.add_row(['' for _ in range(5)])
# ------------------------------------------------------------------

# ------------------------------ 方法 ------------------------------
flag = '🧊 方法'
ptab.add_row([flag, 'p.absolute()', p.absolute(), type(p.absolute()), '返回对象的绝对路径'])
ptab.add_row([flag, 'p.as_posix()', p.as_posix(), type(p.as_posix()), '返回路径的POSIX风格字符串表示'])
ptab.add_row(['📖 注释', '', 'OSIX路径字符串主要在Unix-like系统（如Linux和macOS）中使用', '', '它们以/作为路径分隔符'])
ptab.add_row([flag, 'p.as_uri()', p.as_uri(), type(p.as_uri()), '返回路径的文件URI表示（💡 如果创建p为相对路径则报错）'])
ptab.add_row([flag, 'p.chmod()', p.chmod(0o744), type(p.chmod(0o744)), '改变文件的模式和权限位（💡 如果文件不存在则报错）'])
ptab.add_row([flag, 'p.cwd()', p.cwd(), type(p.cwd()), '返回当前工作目录（绝对路径）'])
ptab.add_row([flag, 'p.expanduser()', p.expanduser(), type(p.expanduser()), '展开路径中的~和~user'])
ptab.add_row(['📖 注释', 'expanduser()方法只对', '以~开头的路径有效。如果路径中不包含~，那么调用expanduser()方法', '不会有任何效果', '~通常代表/home/username'])
ptab.add_row([flag, 'p.home()', p.home(), type(p.home()), '返回当前用户的主目录'])
ptab.add_row([flag, 'p.is_absolute()', p.is_absolute(), type(p.is_absolute()), '判断当前路径是否为绝对路径'])
ptab.add_row([flag, 'p.is_dir()', p.is_dir(), type(p.is_dir()), '判断当前路径是否为一个文件夹📂'])
ptab.add_row([flag, 'p.is_file()', p.is_file(), type(p.is_file()), '判断当前路径是否为一个文件📑'])
ptab.add_row([flag, 'd.iterdir()', [dir.name for dir in list(d.iterdir())], type(d.iterdir()), '迭代目录中的所有路径（💡 如果不是一个目录则报错）'])
ptab.add_row([flag, 'd.join(str, str)', d.joinpath('123', 'abc', '结束'), type(d.joinpath('123', 'abc', '结束')), '连接两个或多个路径'])
ptab.add_row([flag, 'd.mkdir()', d.mkdir(parents=False, exist_ok=True), type(d.mkdir(parents=False, exist_ok=True)), '创建目录（💡 有两个报错参数！）'])
ptab.add_row(['📖 注释', 'mode=511', 'parent=False', 'exist_ok=False', 'os.mkdir和os.makedirs的结合体'])
ptab.add_row([flag, 'f.relative_to(base_path)', f.relative_to('Datasets/coco128'), type(f.relative_to(base_path)), '计算相对路径（💡 需提供基准路径）'])

print(ptab)

# .open()方法
with f.open('r', encoding='utf-8') as file:
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    print(f"{lines = }")
    # lines = ['0 0.497506 0.514227 0.340304 0.846516', '0 0.316557 0.668648 0.30178 0.541047', '28 0.838735 0.765313 0.322529 0.346531']


# ------------------------------------------------------------------