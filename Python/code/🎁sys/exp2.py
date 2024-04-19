import sys

for idx, item in enumerate(sys.argv, start=0):
    print(f"[参数-{idx}] {item} (type={type(item)})")