"""
+ 脚本说明：根据帧间隔对某个文件夹下指定类型的视频文件进行抽帧，得到系列图片。
  + 视频文件所在文件夹名称: `EXAMPLE_FOLDER`
  + 抽帧得到的文件夹名称: `EXAMPLE_FOLDER/extract_frames_results/test_vid_0001.jpg`
+ 用途：将拍摄得到的视频转换为常用的数据集
+ 要求：无
"""
import cv2
import os
import tqdm
from utils import create_folder
import tabulate


"""============================ 需要修改的地方 ==================================="""
SRC_PATH = "Python/常用脚本/EXAMPLE_FOLDER"  # 原始视频路径
frame_interval = 10  # 视频采样间隔，越小采样率越高 -> 60 | 30 | 15 | 10
video_type = ['.mp4', '.avi']  # 视频格式(.mp4 | .avi)

DST_PATH = f"extract_frames_results-{frame_interval}"  # 保存图片文件夹名称
save_img_format = '.jpg'  # 保存的图片格式(.jpg | .png)
"""==============================================================================="""

# 构建路径
results_imgs_path = os.path.join(SRC_PATH, DST_PATH)  # 保存图片路径

# 得到存放所有视频的list
video_list = [x for x in os.listdir(SRC_PATH) if os.path.splitext(x)[-1] in video_type]

"------------计数------------"
TOTAL_VID_NUM = len(video_list)
SUCCEED_NUM = 0  # 完成视频的个数
TOTAL_IMG_NUM = 0  # 统计得到的所有图片数量
"---------------------------"

def calculate_video_duration(video_list):
    total_duration = 0

    # 使用tqdm库创建一个进度条
    for file_name in tqdm.tqdm(video_list, desc='计算视频时长', unit='个'):
        file_path = os.path.join(SRC_PATH, file_name)
        
        # 使用OpenCV读取视频文件
        video_capture = cv2.VideoCapture(file_path)  
        
        # 获取视频的帧数和帧率
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = video_capture.get(cv2.CAP_PROP_FPS)

        # 计算视频时长（单位：秒）
        duration = frame_count / frame_rate
        total_duration += duration  # 累加到总时长
        
        video_capture.release()  # 释放视频捕获对象
    return total_duration

# 调用函数计算总时长
total_duration = calculate_video_duration(video_list)

_str = [
    ["[SRC]视频路径为", SRC_PATH],
    ["视频个数", TOTAL_VID_NUM],
    ["视频总时长", f"{total_duration:.2f} 秒"],
    ['',''],
    ["[DST]图片保存路径为", results_imgs_path],
    ["保存的图片格式为", save_img_format],
]

_str = tabulate.tabulate(_str, headers=["Key", "Value"], tablefmt='pipe')

print(f"\n{_str}\n\n"
      f"请输入 'yes' 继续，输入其他停止")

_INPUT = input()
if _INPUT != "yes":
    exit()
    
# 创建文件夹
if not os.path.exists(results_imgs_path):
    os.mkdir(results_imgs_path)

# 创建一个tqdm进度条对象
progress_bar = tqdm.tqdm(total=len(video_list), desc="视频拆帧...", unit="vid")
statistics_dict = dict()  # 创建一个字典，用于统计
for vid_name in video_list:  # 遍历所有的视频
    save_number = 1  # 记录当前视频保存的frame个数
    vid_pre, vid_ext = os.path.splitext(vid_name)  # 获取文件名和后缀
    
    vid_path = os.path.join(SRC_PATH, vid_name)  # 视频完整路径
    
    # 创建VideoCapture对象
    vc = cv2.VideoCapture(vid_path)

    # 检查视频是否成功打开
    if not vc.isOpened():
        continue
    
    # 逐帧读取视频并保存为图片
    frame_count = 0
    while True:
        # 读取一帧
        rval, frame = vc.read()

        # 检查是否成功读取帧
        if not rval:  # 读取帧失败
            break

        # 每隔 frame_interval 帧保存一次图片
        if frame_count % frame_interval == 0:
            # 生成图片文件名
            frame_name = f"{vid_pre}_{save_number:04d}{save_img_format}"
            frame_path = os.path.join(results_imgs_path, frame_name)  # Python\常用脚本\EXAMPLE_FOLDER\extract_frames_results\test_vid_0016.jpg

            progress_bar.set_description(f"\033[1;31m{vid_name}\033[0m -> "
                                            f"\033[1;36m{save_number * frame_interval:04d}\033[0m"
                                            f" ({save_number})")  # 更新tqdm的描述
            # 保存帧为图片
            cv2.imwrite(frame_path, frame)
            save_number += 1

        # 帧数加1
        frame_count += 1

    # 释放VideoCapture对象
    vc.release()
    TOTAL_IMG_NUM += save_number  # 更新图片数量
    SUCCEED_NUM += 1
    statistics_dict[vid_pre] = save_number  # 更新字典，记录当前视频得到的frame个数
    progress_bar.update()  
progress_bar.close()

_str = []
_cont = 1
for k, v in statistics_dict.items():
    _str.append([_cont, k, v])  # 序号 | 视频名称 | 得到图片数量
    _cont += 1

_str.append(['', f"视频拆帧 ({TOTAL_VID_NUM}个)完成", f"得到[{TOTAL_IMG_NUM}]张[{save_img_format}]图片!"])
_str.append(['', "结果保存路径", results_imgs_path])

_str = tabulate.tabulate(_str, headers=['No', 'Video Name', 'Obtained Images Number'])
print(_str)
