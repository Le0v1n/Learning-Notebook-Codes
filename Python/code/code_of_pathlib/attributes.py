from pathlib import Path


dataset = Path('/mnt/f/Learning-Notebook-Codes/Datasets')
dataset_compressed_package = Path('/mnt/f/Learning-Notebook-Codes/Datasets.tar.gz')
image = Path('Datasets/coco128/train/images/000000000061.jpg')
label = Path('Datasets/coco128/train/labels/000000000061.txt')

print(f"[{dataset.name}] {dataset.parts}")
print(f"[{dataset_compressed_package.name}] {dataset_compressed_package.parts}")
print(f"[{image.name}] {image.parts}")
print(f"[{label.name}] {label.parts}")
