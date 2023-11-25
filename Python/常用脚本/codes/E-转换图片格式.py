"""
+ 脚本说明：对指定文件夹下所有的图片进行格式转换
+ 用途：统一数据集图片的格式
+ 要求：无
+ 注意：
  1. in-place操作
  2. 不需要转换的则跳过
  3. 不是图片的文件会扔到指定位置 RECYCLE_BIN_PATH
"""
import os
import tqdm
from PIL import Image
import shutil


"""============================ 需要修改的地方 ==================================="""
# 定义文件夹路径
IMG_PATH                 = "EXAMPLE_FOLDER/images"  # 输入图片所在文件夹路径
wanna_convert_image_type = '.jpg'  # 想要转换的图片格式
other_image_type         = ['.png', '.jpeg']  # 什么格式的图片将会被转换
"""==============================================================================="""

# 确定回收站位置
RECYCLE_BIN_PATH = os.path.join(os.path.dirname(IMG_PATH), "recycle_bin")

# 获取文件夹内所有文件
all_files = os.listdir(IMG_PATH)

"------------计数------------"
TOTAL_NUM           = len(all_files)
SUCCEED_CONVERT_NUM = 0
SKIP_CONVERT_NUM    = 0
OTHER_FILE_NUM      = 0
"---------------------------"

# 遍历所有的图片
process_bar = tqdm.tqdm(total=TOTAL_NUM, desc=f"将所有图片转换为{wanna_convert_image_type}格式", unit='file')
for file_name in all_files:
    # 分离文件名和后缀
    file_pre, file_ext = os.path.splitext(file_name)
    process_bar.set_description(f"Process in \033[1;31m{file_name}\033[0m")
    
    # 构建文件完整路径
    file_path = os.path.join(IMG_PATH, file_name)
    
    # 检查文件是否为.jpg格式
    if file_ext == wanna_convert_image_type:  # 如果是 jpg 则跳过
        SKIP_CONVERT_NUM += 1
        process_bar.update()
        continue
    elif file_ext in other_image_type:  # 如果是其他图片格式
        with Image.open(file_path) as img:
            # 构建输出文件路径
            dst_save_path = os.path.join(IMG_PATH, file_pre) + wanna_convert_image_type
            img.save(dst_save_path)  # 保存为.jpg格式
            
            # 将原有的图片移动到其他文件夹下
            dst_move_path = os.path.join(RECYCLE_BIN_PATH, file_name)
            shutil.move(src=file_path, dst=dst_move_path)

            SUCCEED_CONVERT_NUM += 1
            process_bar.update()
    else:  # 既不是 jpg 也不是 png、jpeg，则移动到其他文件夹下
        if not os.path.exists(RECYCLE_BIN_PATH):
            os.mkdir(RECYCLE_BIN_PATH)
            
        dst_move_path = os.path.join(RECYCLE_BIN_PATH, file_name)
        shutil.move(src=file_path, dst=dst_move_path)
        OTHER_FILE_NUM += 1
        process_bar.update()
process_bar.close()

print(f"👌 所有图片已转换为jpg, 详情如下:"
      f"\n\t成功转换数量/总文件数量 = \033[1;32m{SUCCEED_CONVERT_NUM}\033[0m/{TOTAL_NUM}"
      f"\n\t跳过文件数量/总文件数量 = \033[1;34m{SKIP_CONVERT_NUM}\033[0m/{TOTAL_NUM}"
      f"\n\t其他格式文件数量/总文件数量 = \033[1;31m{OTHER_FILE_NUM}\033[0m/{TOTAL_NUM}")

if SUCCEED_CONVERT_NUM + SKIP_CONVERT_NUM + OTHER_FILE_NUM == TOTAL_NUM:
    print("👌 No Problems")
else:
    print(f"🤡 貌似有点问题, 请仔细核查!"
          f"\n\tSUCCEED_NUM: {SUCCEED_CONVERT_NUM}"
          f"\n\tSKIP_NUM: {SKIP_CONVERT_NUM}"
          f"\n\tOTHER_FILE_NUM = {OTHER_FILE_NUM}"
          f"\nSUCCEED_NUM + SKIP_NUM + OTHER_FILE_NUM = {SUCCEED_CONVERT_NUM + SKIP_CONVERT_NUM + OTHER_FILE_NUM}"
          f"\nTOTAL_NUM: {TOTAL_NUM}")