import os


iter_num = 0
for dirpath, dirnames, filenames in os.walk('Python/code'):
    print(f"-------------------------- {iter_num = } --------------------------")
    print(f"{dirpath = }")
    print(f"{dirnames = }")
    print(f"{filenames = }")
    iter_num += 1