from pathlib import Path
import os


def get_path_type_Path(path: Path) -> str:
    if path.is_file():
        return 'file'
    elif path.is_dir():
        return 'dir'
    else:
        return 'unknown'
    
    
def get_path_type_os(path: str) -> str:
    if os.path.isfile(path):
        return 'file'
    elif os.path.isdir(path):
        return 'dir'
    else:
        return 'unknown'
            

dataset = Path('/mnt/f/Learning-Notebook-Codes/Datasets')
dataset_compressed_package = Path('/mnt/f/Learning-Notebook-Codes/Datasets.tar.gz')
image = Path('Datasets/coco128/train/images/000000000061.jpg')
label = Path('Datasets/coco128/train/labels/000000000061.txt')

print(f"[{dataset.name}] [调用os库] {get_path_type_os(dataset.absolute()) = }")
print(f"[{dataset.name}] {get_path_type_Path(dataset) = }")
print(f"[{dataset_compressed_package.name}] [调用os库] {get_path_type_os(dataset_compressed_package.absolute()) = }")
print(f"[{dataset_compressed_package.name}] {get_path_type_Path(dataset_compressed_package) = }")
print(f"[{image.name}] [调用os库] {get_path_type_os(image.absolute()) = }")
print(f"[{image.name}] {get_path_type_Path(image) = }")
print(f"[{label.name}] [调用os库] {get_path_type_os(label.absolute()) = }")
print(f"[{label.name}] {get_path_type_Path(label) = }")
