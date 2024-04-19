import os


wanna_create_folder_path = 'Python/code'
if not os.path.exists(wanna_create_folder_path):
    os.mkdir(wanna_create_folder_path)
    print(f"✅  文件夹不存在，已创建！")
else:
    print(f"⚠️  文件夹已经存在，无需创建！")