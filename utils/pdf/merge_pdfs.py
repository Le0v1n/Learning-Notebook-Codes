import os
from pypdf import PdfWriter

merger = PdfWriter()

path_dir = 'pdf文件所在文件夹'

# 按顺序放置pdf文件名
files_list = [
    'file1.pdf',
    'file2.pdf',
]

files_list = [os.path.join(path_dir, file) for file in files_list]

for pdf in files_list:
    merger.append(pdf)
    
save_path = os.path.join(path_dir, "merged-pdf.pdf")

merger.write(save_path)
merger.close()

print(f"Output path: {save_path}")
