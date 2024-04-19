import os


for path, folder_list, file_list in os.walk('Python/code/📂folder1'):
    print(f"{path = }")
    print(f"{folder_list = }")
    print(f"{file_list = }")