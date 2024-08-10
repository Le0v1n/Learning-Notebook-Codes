from pathlib import Path
from PIL import Image

def png2jpg(src_dir: str, dst_dir: str):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not src_dir.exists():
        raise FileNotFoundError

    # 遍历源目录中的所有文件
    for png_filepath in src_dir.glob('*.png'):
            with Image.open(png_filepath) as img:
                jpg_filepath = dst_dir.joinpath(png_filepath.stem + '.jpg')
                # 转换格式并保存为JPG
                img.convert('RGB').save(jpg_filepath, 'JPEG')

    print(f'转换完成，JPG文件已保存到 {dst_dir}')


if __name__ == "__main__":
    source_directory = 'test'  # PNG文件所在的目录
    output_directory = 'test'  # 输出JPG文件的目录

    png2jpg(source_directory, output_directory)