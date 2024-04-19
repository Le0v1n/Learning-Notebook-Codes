import os


def get_file_size(filepath, unit='MB', ndigits=4):
    """获取文件大小
    Args:
        fp (str): 文件路径
        unit (str): 单位选项，可以是'KB', 'MB', 'GB'等
        ndigits (int): 小数点后保留的位数
    Returns:
        float: 文件大小(默认为MB)
    """
    
    # 获取文件大小（字节）
    file_size_bytes = os.path.getsize(filepath)
    unit = unit.upper()
    
    # 单位到字节倍数的映射
    unit_multipliers = {
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024,
    }
    
    # 根据单位转换文件大小
    if unit in unit_multipliers:
        multiplier = unit_multipliers[unit]
        file_size = round(file_size_bytes / multiplier, ndigits=ndigits)
    else:
        # 默认或未知单位时使用MB
        file_size = round(file_size_bytes / (1024 * 1024), ndigits=ndigits)
        unit = 'MB'
    return file_size


if __name__ == '__main__':
    filepath = 'Python/Python中的os模块和sys模块.md'
    filesize = get_file_size(filepath=filepath, unit='MB', ndigits=4)
    print(f"{filesize = } MB")
    