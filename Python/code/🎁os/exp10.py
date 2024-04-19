import os
import warnings


filepath = 'Python/code/exp100000.py'
if not os.path.exists(filepath):
    warnings.warn(f"⚠️  文件 {filepath} 并不存在！")
else:
    ...