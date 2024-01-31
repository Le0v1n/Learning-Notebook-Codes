import time
from tqdm.rich import tqdm

# 方法1
obj = list(range(20))
for i in tqdm(obj, desc="Test", total=len(obj)):
    time.sleep(0.1)
    
# 方法2
progress_bar = tqdm(total=len(obj), desc="Test2")
for i in obj:
    progress_bar.set_description(f"{i = }")
    time.sleep(0.1)
    progress_bar.update()
progress_bar.close()

progress_bar = tqdm(total=len(obj), desc="Test2")
for i in range(200):
    progress_bar.set_description(f"{i = }")
    time.sleep(0.05)
    progress_bar.update()
progress_bar.close()
