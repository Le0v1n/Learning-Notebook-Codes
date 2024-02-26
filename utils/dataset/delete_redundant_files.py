import os
import sys
import tqdm

# 将项目路径添加到pwd中
sys.path.append(os.getcwd())

from utils.common_fn import print_arguments, xprint

__doc__ = """脚本说明：
    Functions：根据文件夹A删除冗余的（不匹配A）的文件夹B。

    Example：假如文件夹A中存放的是图片，文件夹B存放的是对应的json标签。
             运行下面的脚本，则会将文件夹B中没有和文件夹A匹配的标签删除掉。
"""


"""============================ 需要修改的地方 ==================================="""
path_A = 'Datasets/coco128/train/images'  # 肯定不会被删除
file_type_A = ('.jpg', '.png')  # 注意格式！

path_B = 'Datasets/coco128/train/labels'  # ⚠️可能被删除
file_type_B = ('.txt', '.json', '.xml')  # 注意格式
"""==============================================================================="""

# 获取两种文件列表
files_A_list = [file for file in os.listdir(path_A) if file.endswith(file_type_A)]
files_B_list = [file for file in os.listdir(path_B) if file.endswith(file_type_B)]

"------------计数------------"
NUM_A = len(files_A_list)
NUM_B = len(files_B_list)
SUCCEED_NUM = 0
SKIP_NUM = 0
"---------------------------"

xprint(__doc__, color='blue')
print_arguments(文件夹A的路径为=path_A, 
                文件夹A的文件后缀为=file_type_A, 
                文件夹A的文件数量为=NUM_A,
                文件夹B的路径为=path_B,
                文件夹B的文件后缀为=file_type_B,
                文件夹B的文件数量为=NUM_B,
                wait=True)

# 遍历文件B
process_bar = tqdm.tqdm(total=NUM_B, desc="根据文件A删除冗余的文件B", unit='unit')
for name_B in files_B_list:
    pre_B, ext_A = os.path.splitext(name_B)  # 分离文件名和后缀
    process_bar.set_description(f"Process with \033[1;31m{name_B}\033[0m")
    
    # 判断对应的同名 A 文件是否存在，如果存在则跳过
    dst_path = os.path.join(path_A, pre_B)  # 没有后缀
    _exist_flag = 0
    for ext_A in file_type_A:  # 遍历所有格式，看是否有至少一个同名文件存在
        if os.path.exists(dst_path + ext_A):
            _exist_flag += 1
    if _exist_flag > 0:  # 如果存在至少一个同名文件, 则跳过
        SKIP_NUM += 1
        process_bar.update()
    else:  # 没有同名文件, 则删除文件B
        del_path = os.path.join(path_B, name_B)
        os.remove(del_path)
        SUCCEED_NUM += 1
        
        process_bar.update()
        
process_bar.close()

# 统计结果
print_arguments(
    "👌 冗余的B文件删除已完成!",
    删除文件数量_文件B数量=(SUCCEED_NUM, NUM_B),
    跳过文件数量_文件B数量=(SKIP_NUM, NUM_B)
)

if SUCCEED_NUM + SKIP_NUM == NUM_B:
    xprint("✔️  No Problems", color='red', bold=True, hl=">", hl_style='full')
else:
    xprint("❌  有问题，请仔细核对!", color='red', bold=True, hl=">", hl_style='full')
    print_arguments(
        SUCCEED_NUM=SUCCEED_NUM,
        SKIP_NUM=SKIP_NUM,
        SUCCEED_NUM_SKIP_NUM_ERROR_NUM=SUCCEED_NUM+SKIP_NUM,
        TOTAL_NUM=NUM_B
    )