import os


def get_file_size(file_path):
    # 获取文件大小（字节）
    file_size_bytes = os.path.getsize(file_path)
    
    # 将字节转换为MB
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    return file_size_mb