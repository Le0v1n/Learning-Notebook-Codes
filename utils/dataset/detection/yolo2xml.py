"""
+ 脚本说明：目标检测中yolo标注文件转换为xml格式
+ 用途：YOLO 模型推理得到 txt 文件 -> 转换为 xml 标注文件。
+ 要求：要有对应的图片文件，这样读取到的尺寸信息是最准确的。
"""
from xml.dom.minidom import Document
import os
import cv2
import tqdm


"""============================ 需要修改的地方 ==================================="""
IMAGE_PATH = "Datasets/coco128/train/images"  # 原图文件夹路径
TXT_PATH = "Datasets/coco128/train/labels"  # 原txt标签文件夹路径
XML_PATH = "Datasets/coco128/train/annotations-xml"  # 保存xml文件夹路径
classes_dict = {  # 🧡类别字典
    '0': 'person',
    '1': 'bicycle',
}

image_type = '.jpg'
create_empty_xml_for_neg = False  # 是否为负样本生成对应的空的xml文件
"""==============================================================================="""

# 读取所有的.txt文件
txt_file_list = [file for file in os.listdir(TXT_PATH) if file.endswith("txt") and file != 'classes.txt']

"------------计数------------"
TOTAL_NUM = len(txt_file_list)
SUCCEED_NUM = 0  # 成功创建xml数量
SKIP_NUM = 0  # 跳过创建xml文件数量
OBJECT_NUM = 0  # object数量
ERROR_NUM = 0  # 没有对应图片
"---------------------------"

_str = (f"💡 图片路径: \033[1;33m{IMAGE_PATH}\033[0m"
        f"\n💡 TXT文件路径为: \033[1;33m{TXT_PATH}\033[0m"
        f"\n💡 JSON文件路径为: \033[1;33m{XML_PATH}\033[0m"
        f"\n 所有TXT文件数量: \033[1;33m{TOTAL_NUM}\033[0m"
        f"\n 类别字典为:")

for idx, value in classes_dict.items():
    _str += f"\n\t[{idx}] {value}"

_str += f"\n\n请输入 \033[1;31m'yes'\033[0m 继续，输入其他停止"
print(_str)

_INPUT = input()
if _INPUT != "yes":
    exit()

os.makedirs(XML_PATH) if not os.path.exists(XML_PATH) else None

process_bar = tqdm.tqdm(total=TOTAL_NUM, desc="yolo2xml", unit='.txt')
for i, txt_name in enumerate(txt_file_list):
    process_bar.set_description(f"Process in \033[1;31m{txt_name}\033[0m")
    txt_pre, txt_ext = os.path.splitext(txt_name)  # 分离前缀和后缀
    
    xmlBuilder = Document()  # 创建一个 XML 文档构建器
    annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
    xmlBuilder.appendChild(annotation)
    
    # 打开 txt 文件
    txtFile = open(os.path.join(TXT_PATH, txt_name))
    txtList = txtFile.readlines()  # 以一行的形式读取txt所有内容
    
    if not txtList and not create_empty_xml_for_neg:  # 如果 txt 文件内容为空且不允许为负样本创建xml文件
        SKIP_NUM += 1
        process_bar.update()
        continue
    
    # 读取图片
    if not os.path.exists(os.path.join(IMAGE_PATH, txt_pre) + image_type):
        ERROR_NUM += 1
        process_bar.update()
        continue
    img = cv2.imread(os.path.join(IMAGE_PATH, txt_pre) + image_type)
    H, W, C = img.shape
    
    # folder标签
    folder = xmlBuilder.createElement("folder")  
    foldercontent = xmlBuilder.createTextNode('images')
    folder.appendChild(foldercontent)
    annotation.appendChild(folder)  # folder标签结束

    # filename标签
    filename = xmlBuilder.createElement("filename")  
    filenamecontent = xmlBuilder.createTextNode(txt_pre + image_type)
    filename.appendChild(filenamecontent)
    annotation.appendChild(filename)  # filename标签结束

    # size标签
    size = xmlBuilder.createElement("size")  
    width = xmlBuilder.createElement("width")  # size子标签width
    widthcontent = xmlBuilder.createTextNode(str(W))
    width.appendChild(widthcontent)
    size.appendChild(width)  # size子标签width结束

    height = xmlBuilder.createElement("height")  # size子标签height
    heightcontent = xmlBuilder.createTextNode(str(H))
    height.appendChild(heightcontent)
    size.appendChild(height)  # size子标签height结束

    depth = xmlBuilder.createElement("depth")  # size子标签depth
    depthcontent = xmlBuilder.createTextNode(str(C))
    depth.appendChild(depthcontent)
    size.appendChild(depth)  # size子标签depth结束
    annotation.appendChild(size)  # size标签结束
    
    # 读取 txt 内容，生成 xml 文件内容
    for line in txtList:  # 正样本(txt内容不为空)
        # .strip()去除行首和行尾的空白字符（如空格和换行符）
        oneline = line.strip().split(" ")  # oneline是一个list, e.g. ['0', '0.31188484251968507', 
                                           #                         '0.6746135899679205', 
                                           #                         '0.028297244094488208', 
                                           #                         '0.04738990959463407']

        # 开始 object 标签
        object = xmlBuilder.createElement("object")  # object 标签
        
        # 1. name标签
        picname = xmlBuilder.createElement("name")  
        namecontent = xmlBuilder.createTextNode(classes_dict[oneline[0]])  # 确定是哪个类别
        picname.appendChild(namecontent)
        object.appendChild(picname)  # name标签结束

        # 2. pose标签
        pose = xmlBuilder.createElement("pose")  
        posecontent = xmlBuilder.createTextNode("Unspecified")
        pose.appendChild(posecontent)
        object.appendChild(pose)  # pose标签结束

        # 3. truncated标签
        truncated = xmlBuilder.createElement("truncated")  
        truncatedContent = xmlBuilder.createTextNode("0")
        truncated.appendChild(truncatedContent)
        object.appendChild(truncated)  # truncated标签结束
        
        # 4. difficult标签
        difficult = xmlBuilder.createElement("difficult")  
        difficultcontent = xmlBuilder.createTextNode("0")
        difficult.appendChild(difficultcontent)
        object.appendChild(difficult)  # difficult标签结束

        # 5. bndbox标签
        bndbox = xmlBuilder.createElement("bndbox")  
        ## 5.1 xmin标签
        xmin = xmlBuilder.createElement("xmin")  
        mathData = float(((float(oneline[1])) * W) - (float(oneline[3])) * 0.5 * W)
        xminContent = xmlBuilder.createTextNode(str(mathData))
        xmin.appendChild(xminContent)
        bndbox.appendChild(xmin)  # xmin标签结束

        ## 5.2 ymin标签
        ymin = xmlBuilder.createElement("ymin")  # ymin标签
        mathData = float(((float(oneline[2])) * H) - (float(oneline[4])) * 0.5 * H)
        yminContent = xmlBuilder.createTextNode(str(mathData))
        ymin.appendChild(yminContent)
        bndbox.appendChild(ymin)  # ymin标签结束
        
        ## 5.3 xmax标签
        xmax = xmlBuilder.createElement("xmax")  # xmax标签
        mathData = float(((float(oneline[1])) * W) + (float(oneline[3])) * 0.5 * W)
        xmaxContent = xmlBuilder.createTextNode(str(mathData))
        xmax.appendChild(xmaxContent)
        bndbox.appendChild(xmax)  # xmax标签结束

        ## 5.4 ymax标签
        ymax = xmlBuilder.createElement("ymax")  # ymax标签
        mathData = float(
            ((float(oneline[2])) * H) + (float(oneline[4])) * 0.5 * H)
        ymaxContent = xmlBuilder.createTextNode(str(mathData))
        ymax.appendChild(ymaxContent)
        bndbox.appendChild(ymax)  # ymax标签结束

        object.appendChild(bndbox)  # bndbox标签结束
        annotation.appendChild(object)  # object标签结束
        
        OBJECT_NUM += 1

    # 创建 xml 文件
    f = open(os.path.join(XML_PATH, txt_pre) + '.xml', 'w')

    # 为 创建好的 xml 文件写入内容
    xmlBuilder.writexml(f, indent='\t', newl='\n',
                        addindent='\t', encoding='utf-8')
    f.close()  # 关闭xml文件
    
    SUCCEED_NUM += 1
    process_bar.update()
process_bar.close()

print(f"👌yolo2xml已完成, 详情如下:"
      f"\n\t成功转换文件数量/总文件数量 = \033[1;32m{SUCCEED_NUM}\033[0m/{TOTAL_NUM}"
      f"\n\t跳过转换文件数量/总文件数量 = \033[1;31m{SKIP_NUM}\033[0m/{TOTAL_NUM}"
      f"\n\t所有样本的 object 数量/总文件数量 = \033[1;32m{OBJECT_NUM}\033[0m/{TOTAL_NUM}"
      f"\n\t平均每个xml文件中object的数量为: {int(OBJECT_NUM / SUCCEED_NUM)}"
      f"\n\t没有对应图片的数量为: {ERROR_NUM}"
      f"\n\t结果保存路径为: {XML_PATH}")

if SUCCEED_NUM + SKIP_NUM == TOTAL_NUM:
    print(f"\n👌 \033[1;32mNo Problem\033[0m")
else:
    print(f"\n🤡 \033[1;31m貌似有点问题, 请仔细核查!\033[0m")