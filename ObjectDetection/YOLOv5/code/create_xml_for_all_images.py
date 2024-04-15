import os
import xml.dom.minidom
from tqdm import tqdm


# 为哪些图片生成 .xml 文件？
img_path = '/mnt/c/Users/Le0v1n/Desktop/测试案例/Datasets/exp_1/JPEGImages'

# 将生成的 .xml 文件保存到哪个文件夹下？
xml_path = '/mnt/c/Users/Le0v1n/Desktop/测试案例/Datasets/exp_1/Empty_Annotations'

# 获取图像文件列表
img_files = os.listdir(img_path)

# 使用tqdm创建进度条
for img_file in tqdm(img_files, desc="生成XML文件"):
    img_name = os.path.splitext(img_file)[0]

    # 创建一个空的DOM文档对象
    doc = xml.dom.minidom.Document()
    # 创建名为annotation的根节点
    annotation = doc.createElement('annotation')
    # 将根节点添加到DOM文档对象
    doc.appendChild(annotation)

    # 添加folder子节点
    folder = doc.createElement('folder')
    folder_text = doc.createTextNode('VOC2007')
    folder.appendChild(folder_text)
    annotation.appendChild(folder)

    # 添加filename子节点
    filename = doc.createElement('filename')
    filename_text = doc.createTextNode(img_file)
    filename.appendChild(filename_text)
    annotation.appendChild(filename)

    # 添加path子节点
    path = doc.createElement('path')
    path_text = doc.createTextNode(img_path + '/' + img_file)  # 修正路径
    path.appendChild(path_text)
    annotation.appendChild(path)

    # 添加source子节点
    source = doc.createElement('source')
    database = doc.createElement('database')
    database_text = doc.createTextNode('Unknown')
    source.appendChild(database)
    database.appendChild(database_text)
    annotation.appendChild(source)

    # 添加size子节点
    size = doc.createElement('size')
    width = doc.createElement('width')
    width_text = doc.createTextNode('1280')
    height = doc.createElement('height')
    height_text = doc.createTextNode('720')
    depth = doc.createElement('depth')
    depth_text = doc.createTextNode('3')
    size.appendChild(width)
    width.appendChild(width_text)
    size.appendChild(height)
    height.appendChild(height_text)
    size.appendChild(depth)
    depth.appendChild(depth_text)
    annotation.appendChild(size)

    # 添加segmented子节点
    segmented = doc.createElement('segmented')
    segmented_text = doc.createTextNode('0')
    segmented.appendChild(segmented_text)
    annotation.appendChild(segmented)

    # 将XML写入文件
    xml_file_path = os.path.join(xml_path, f'{img_name}.xml')
    with open(xml_file_path, 'w+', encoding='utf-8') as fp:
        doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding='utf-8')