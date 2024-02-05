import os
import tqdm


"""============================ 需要修改的地方 ==================================="""
path_A = 'Datasets/coco128/train/images'  # 肯定不会被删除
file_type_A = ('.jpg', '.png')

path_B = 'Datasets/coco128/train/labels'  # 可能被删除
file_type_B = ('.txt', '.json', '.xml')
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

print(f"文件[A]所在文件夹路径为: {path_A}"
      f"\n\t文件[A]数量为: {NUM_A}"
      f"\n\t文件[A]的后缀为: {file_type_A}"
      f"\n文件[B]所在文件夹路径为: {path_B}"
      f"\n\t文件[B]数量为: {NUM_B}"
      f"\n\t文件[B]的后缀为: {file_type_B}"
      f"\n\n请输入 \033[1;31m'yes'\033[0m 继续，输入其他停止")
_INPUT = input()
if _INPUT != "yes":
    exit()

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

print(f"👌 冗余的B文件删除已完成!"
      f"\n\t删除文件数量/文件B数量 = \033[1;31m{SUCCEED_NUM}\033[0m/{NUM_B}"
      f"\n\t跳过文件数量/文件B数量 = \033[1;32m{SKIP_NUM}\033[0m/{NUM_B}")

if SUCCEED_NUM + SKIP_NUM == NUM_B:
    print("👌 No Problems")
else:
    print(f"🤡 有问题，请仔细核对!"
          f"\n\tSUCCEED_NUM: {SUCCEED_NUM}\tSKIP_NUM: {SKIP_NUM}"
          f"\n\tSUCCEED_NUM + SKIP_NUM + ERROR_NUM = {SUCCEED_NUM + SKIP_NUM}"
          f"\n\tTOTAL_NUM: {NUM_B}")