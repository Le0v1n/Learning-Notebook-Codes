from tqdm import tqdm as TQDM
from pathlib import Path
from typing import Union, Tuple, List
from utils.items import ImageFormat, LabelFormat, VideoFormat


def fix_suffix(suffixs: Union[str, list]) -> list:
    if isinstance(suffixs, str):
        suffixs = [suffixs, ]

    results: list = []
    for suffix in suffixs:
        suffix = '.' + suffix if suffix[0] != '.' else suffix
        results.append(suffix)

    return results


def get_files(dirpath: str, file_type: Union[str, Tuple[str, ...], None] = 'all', path_style: str = 'relative') -> List[str]:
    """
    获取某个文件夹下指定格式的所有文件的路径（不进行递归遍历）。

    Args:
        dirpath (str): 文件夹路径。
        file_type (Union[str, Tuple[str, ...], None], optional): 文件格式，可以是某一格式（如.jpg），也可以是格式的元组，
            或者使用预定义的关键字。
            - 'image'：('.png', '.jpg', '.jpeg', '.gif', '.tiff', '.svg', '.raw', 'webp', '.heic', '.heif')
            - 'annotation'：('.csv', '.json', '.txt', '.xml')
            - 'video'：('.mp4', '.avi', '.mkv', '.wmv', '.flv', '.mov', 'rmvb', '.webm', '.mpg', '.vob')
            - 'all'或'any'：不限制后缀
            - 默认为 'all'。
        path_style (str, optional): 指定返回的文件路径的样式。
            - 'relative'：返回相对路径
            - 'absolute'：返回绝对路径
            - 'filename'：仅返回文件名
            - 默认为 'relative'。

    Returns:
        List[str]: 包含所有指定文件路径的列表。
    """
    assert isinstance(file_type, (str, list, tuple)), \
        f"❌ The type of file_type must be str, list, tuple not {type(file_type)}!"

    dirpath: Path = Path(dirpath)

    if isinstance(file_type, str):
        if file_type.lower() in ('image', 'images', 'pic', 'picture'):  # 图片
            file_type: list = ImageFormat
        elif file_type.lower() in ('annotation', 'annotations', 'label', 'labels'):  # 标签
            file_type: list = LabelFormat
        elif file_type.lower() in ('video', 'videos', 'vid'):  # 标签
            file_type: list = VideoFormat
        elif file_type.lower() in ('all', 'any'):  # 不挑选格式
            file_type = None
        else:
            file_type: list = fix_suffix(file_type)
    
    # ---------- 文件搜索 ----------
    files: list = []
    if file_type:
        for suffix in file_type:
            files.extend(dirpath.glob(f"*{suffix}"))
    else:
        files = list(dirpath.glob('*'))

    # ---------- 删除所有的文件夹 ----------
    files = [file for file in files if not file.is_dir()]
      
    # ---------- 处理path_style ----------
    if path_style.lower() in ('relative', 'rel', ):  # 相对路径
        ...
    elif path_style.lower() in ('absolute', 'abs', ):  # 绝对路径
        files = [filepath.resolve() for filepath in files]
    elif path_style.lower() in ('filename', 'name', ):  # 没有路径
        files = [filepath.name for filepath in files]

    return files


def get_files_recursion(dirpath: Union[str, Path, list], file_type: Union[str, Tuple[str, ...], None] = 'all', 
                        path_style: str = 'relative', exclude_dirpath: list = []) -> List[str]:
    """
    获取某个文件夹下指定格式的所有文件的路径（递归遍历）。

    Args:
        dirpath (str | Path | list): 文件夹路径。
        file_type (Union[str, Tuple[str, ...], None], optional): 文件格式，可以是某一格式（如.jpg），也可以是格式的元组，
            或者使用预定义的关键字。
            - 'image'：('.png', '.jpg', '.jpeg', '.gif', '.tiff', '.svg', '.raw', 'webp', '.heic', '.heif')
            - 'annotation'：('.csv', '.json', '.txt', '.xml')
            - 'video'：('.mp4', '.avi', '.mkv', '.wmv', '.flv', '.mov', 'rmvb', '.webm', '.mpg', '.vob')
            - 'all'或'any'：不限制后缀
            - 默认为 'all'。
        path_style (str, optional): 指定返回的文件路径的样式。
            - 'relative'：返回相对路径
            - 'absolute'：返回绝对路径
            - 'filename'：仅返回文件名
            - 默认为 'relative'。
        exclude_dirpath (list): 需要排除的文件夹的路径（元素是str）

    Returns:
        List[str]: 包含所有指定文件路径的列表。
    """
    # ---------- 处理list的情况 ----------
    files: list = []
    if isinstance(dirpath, (list, tuple, set)):
        for sub_dir in dirpath:
            files.extend(get_files_recursion(sub_dir, file_type, path_style, exclude_dirpath))
        return list(set(files))

    # ---------- 一些必要条件的判断 ----------
    dirpath: Path = Path(dirpath)
    assert dirpath.is_dir(), f"❌ The 'dirpath' must be a directory! -> {dirpath.__str__()}"
    assert isinstance(file_type, (str, list, tuple)), f"❌ The type of file_type must be str, list, tuple not {type(file_type)}!"

    if isinstance(file_type, str):
        if file_type.lower() in ('image', 'images', 'pic', 'picture'):  # 图片
            file_type: list = ImageFormat
        elif file_type.lower() in ('annotation', 'annotations', 'label', 'labels'):  # 标签
            file_type: list = LabelFormat
        elif file_type.lower() in ('video', 'videos', 'vid'):  # 标签
            file_type: list = VideoFormat
        elif file_type.lower() in ('all', 'any'):  # 不挑选格式
            file_type = None
        else:
            file_type: list = fix_suffix(file_type)

    # ---------- 获取所有的文件夹并排除指定的文件夹 ----------
    dirs: list = list(dirpath.rglob('*'))
    dirs = [sub_dir for sub_dir in TQDM(dirs, ascii=' >', desc='Find directory recursion...') \
            if sub_dir.is_dir() and sub_dir.__str__() not in exclude_dirpath]
    
    # ---------- 遍历所有的文件夹，获取指定的文件 ----------
    for sub_dir in TQDM(dirs, ascii=' >', desc="Searching files from directories..."):
        sub_dir: Path
        if file_type:  # 如果有后缀规定
            for suffix in file_type:
                files.extend(sub_dir.rglob(f"*{suffix}"))
        else:  # 没有后缀的要求
            files = list(sub_dir.rglob('*'))

    # ---------- 删除所有的文件夹 ----------
    files = [file for file in files if not file.is_dir()]

    # ---------- 处理path_style ----------
    if path_style.lower() in ('relative', 'rel', ):  # 相对路径
        ...
    elif path_style.lower() in ('absolute', 'abs', ):  # 绝对路径
        files = [filepath.resolve() for filepath in files]
    elif path_style.lower() in ('filename', 'name', ):  # 没有路径
        files = [filepath.name for filepath in files]

    return list(set(files))