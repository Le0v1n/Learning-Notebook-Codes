import os
import sys
import cv2
from tqdm import tqdm
import tabulate
import threading
sys.path.append(os.getcwd())
from utils.generator import create_folder
from utils.outer import print_arguments
from utils.outer import xprint
from utils.items import VideoFormat
from utils.getter import get_files


__doc__ = """脚本说明：根据帧间隔对某个文件夹下指定类型的视频文件进行抽帧，得到系列图片。
    参数:
        视频文件所在文件夹名称: EXAMPLE_FOLDER
        抽帧得到的文件夹名称: EXAMPLE_FOLDER/extract_frames_results/test_vid_0001.jpg
    用途：将拍摄得到的视频转换为常用的数据集
    要求：无
"""
xprint(__doc__, color='blue', bold=True, hl="=", hl_num=2)


"""============================ 需要修改的地方 ==================================="""
videos_dir = ""  # 原始视频路径
frames_save_path = ""  # 保存图片文件夹名称

sample_interval = 15  # 视频采样间隔，越小采样率越高 -> 60 | 30 | 15 | 10
video_default_fps = 25  # [optional] 视频默认的帧数

save_img_format = '.png'  # 保存的图片格式(.jpg | .png)
"""==============================================================================="""

# 得到存放所有视频的list
video_list = get_files(videos_dir, extension=VideoFormat, path_style=None)

"------------计数------------"
count_video_num = len(video_list)
count_finished = 0  # 完成视频的个数
"---------------------------"

def calculate_video_duration(video_list):
    total_duration = 0

    # 使用tqdm库创建一个进度条
    for file_name in tqdm(video_list, desc='计算视频时长', unit='个'):
        file_path = os.path.join(videos_dir, file_name)
        
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

print_arguments(
    视频路径为=videos_dir,
    视频个数=count_video_num,
    视频默认帧率=video_default_fps,
    采样间隔=f"{sample_interval} fps",
    视频总时长=f"{total_duration:.2f} 秒",
    预计数量=f"{total_duration * (video_default_fps / sample_interval):.2f} 秒",
    图片保存路径为=frames_save_path,
    保存的图片格式为=save_img_format,
    wait=True
)

create_folder(frames_save_path, verbose=False)  # 创建文件夹
    
def process_video(vid_name, progress_bar, SUCCEED_NUM, statistics_dict):
    save_number = 1  # 记录当前视频保存的frame个数
    vid_pre, vid_ext = os.path.splitext(vid_name)  # 获取文件名和后缀
    
    vid_path = os.path.join(videos_dir, vid_name)  # 视频完整路径
    
    # 创建VideoCapture对象
    vc = cv2.VideoCapture(vid_path)

    # 检查视频是否成功打开
    if not vc.isOpened():
        return
    
    # 逐帧读取视频并保存为图片
    frame_count = 0
    while True:
        # 读取一帧
        rval, frame = vc.read()

        # 检查是否成功读取帧
        if not rval:  # 读取帧失败
            break

        # 每隔 frame_interval 帧保存一次图片
        if frame_count % sample_interval == 0:
            # 生成图片文件名
            frame_name = f"{vid_pre}_{save_number:05d}{save_img_format}"
            frame_path = os.path.join(frames_save_path, frame_name)  # Python\常用脚本\EXAMPLE_FOLDER\extract_frames_results\test_vid_0016.jpg

            progress_bar.set_description(f"\033[1;31m{vid_name:<30}\033[0m -> "
                                            f"\033[1;36m{save_number * sample_interval:08d}\033[0m"
                                            f" ({save_number:05d})")  # 更新tqdm的描述
            # 保存帧为图片
            cv2.imwrite(frame_path, frame)
            save_number += 1

        # 帧数加1
        frame_count += 1

    # 释放VideoCapture对象
    vc.release()
    SUCCEED_NUM += 1
    statistics_dict[vid_pre] = save_number  # 更新字典，记录当前视频得到的frame个数
    progress_bar.update()  
    

# 创建一个tqdm进度条对象
progress_bar = tqdm(total=len(video_list), desc="视频拆帧...", unit="vid")
statistics_dict = dict()  # 创建一个字典，用于统计
threads = []  # 保存线程的list
for vid_name in video_list:  # 遍历所有的视频
    t = threading.Thread(target=process_video, args=(vid_name, 
                                                     progress_bar, 
                                                     count_finished, 
                                                     statistics_dict))
    threads.append(t)
    t.start()

"""
join()方法的作用是使当前线程等待调用该方法的线程执行完毕后再继续执行。
    由于我们创建了多个线程来处理视频，因此我们需要使用join()方法来等待所有线程都执行完毕后再继续执行主线程
    -> 确保所有线程都执行完毕后再继续执行主线程
"""
for t in threads:
    t.join()

progress_bar.close()

# 由于使用了多线程，因此字典key的顺序可能是乱的，因此根据key重新排序(升序)
sorted_statistics_dict = {}
for key in sorted(statistics_dict.keys()):
    sorted_statistics_dict[key] = statistics_dict[key]
    
_str = []
_cont = 1
_sum = 0  # 统计所有图片数量
for k, v in sorted_statistics_dict.items():
    _str.append([_cont, k, v])  # 序号 | 视频名称 | 得到图片数量
    _cont += 1
    _sum += v  # 统计所有图片数量

_str = tabulate.tabulate(_str, headers=['No', 'Video Name', 'Obtained Images Number'], tablefmt='pretty')
print(_str)

_str = (f"✔️  视频拆帧 ({count_video_num}个)完成! 得到[{_sum}]张[{save_img_format}]图片\n"
        f"结果保存路径为: {frames_save_path}")
xprint(_str, color='green', hl='>')