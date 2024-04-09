import os
import sys
import argparse
import cv2

sys.path.append(os.getcwd())
from utils.getter import get_files
from utils.outer import print_arguments, xprint
from utils.generator import create_folder
from utils.items import fourcc, VideoFormat


def parse_args():
    parser = argparse.ArgumentParser(description="将图片转换为视频")
    parser.add_argument("--images_dir", type=str, default="", help=f"存放图片的文件夹路径")
    parser.add_argument("--save_dir", type=str, default="", help=f"存放结果的文件夹路径")
    parser.add_argument("--fps", type=int, default=24, help=f"生成视频的帧率")
    parser.add_argument("--video_format", type=str, default=".avi", help=f"视频的格式")
    
    args = parser.parse_args()
    return args


def image2video(images_dir: str, save_dir: str, fps: int = 24, video_format: str = '.mp4', comparison_fn=None) -> str:
    """将一个文件夹内的所有图片组合为一个视频
    
    如 ::
    
        images_dir = 'utils/dataset/_example_dataset/jester/10026'
        save_dir = 'utils/dataset/_example_dataset/jester/results'
        fps = 24
        video_format = '.mp4'

    那么会在 utils/dataset/_example_dataset/jester/results 文件夹下生成名为 10026.mp4 的视频文件
    
    - Args:
        - `images_dir (str)`: 文件夹的路径
        - `save_dir (str)`: 要保存到哪个文件夹下
        - `fps (int, optional)`: 生成视频的帧数. Defaults to 24.
        - `video_format (str, optional)`: 视频格式. Defaults to '.mp4'.
        - `comparison_fn (_type_, optional)`: 文件夹内图片的排序规则函数. Defaults to None
            - 当 `comparison_fn is None`: `comparison_fn = lambda i : int(i.split('.')[0])`
            - 如果视频有独特的命名方式，可以自行写一个对比函数

    - Returns:
        - `str`: 保存视频的路径
    """
    video_format = video_format.lower()
    assert video_format in VideoFormat, f"❌  请输入正确的视频格式，如 .mp4，当前为: {video_format}"
    assert isinstance(fps, int), f"❌  请确保 fps 为 int"
    
    # 获取图片的列表
    images_list = get_files(
        fp=images_dir,
        extension='image',
        path_style=None
    )
    
    # 对images_list进行排序
    if not comparison_fn:
        comparison_fn = lambda i : int(i.split('.')[0])
    images_list = sorted(images_list, key=comparison_fn)
    
    # 变为完整路径
    images_list = [os.path.join(images_dir, image_name) for image_name in images_list]
    
    img = cv2.imread(images_list[0])
    size = img.shape[:2][::-1]
    
    if video_format in ('.flv', '.webm', '.mpg', '.vob'):
        xprint(f"⚠️  {video_format} 格式并不能很好的支持!", color='yellow', bold=True)
    
    create_folder(save_dir)
    save_path = os.path.join(save_dir, os.path.basename(images_dir)) + video_format

    video = cv2.VideoWriter(
        filename=save_path, 
        fourcc=fourcc[video_format], 
        fps=fps, 
        frameSize=size
    )
    
    # 视频保存在当前目录下
    for image_path in images_list:
        img = cv2.imread(image_path)
        video.write(img)

    video.release()
    cv2.destroyAllWindows()

    return save_path
    
if __name__ == "__main__":
    args = parse_args()
    print_arguments(argparse=args)

    image2video(
        images_dir=args.images_dir,
        save_dir=args.save_dir,
        fps=args.fps,
        video_format=args.video_format
    )
    