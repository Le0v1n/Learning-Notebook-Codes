"""
+ 脚本说明：目标检测中xml标注文件转换为yolo格式
+ 用途：xml2yolo
+ 要求：要有对应的图片文件，这样读取到的尺寸信息是最准确的。
"""
import os
from lxml import etree
import tqdm
from PIL import Image


"""============================ 需要修改的地方 ==================================="""
IMAGE_PATH = 'EXAMPLE_FOLDER/images'  # 图片路径
XML_PATH = 'EXAMPLE_FOLDER/annotations-xml'  # xml标签路径
TXT_SAVE_PATH = "EXAMPLE_FOLDER/labels"  # yolo标签保存路径

image_type = '.jpg'  # 图片的格式

# 标签从0开始
class_dict = {"cat": 0, 
              "dog": 1}
"""==============================================================================="""

xml_files = [file for file in os.listdir(XML_PATH) if file.lower().endswith('.xml')]

"------------计数------------"
TOTAL_NUM = len(xml_files)
SUCCEED_NUM = 0
NEG_NUM = 0
ERROR_NUM = 0
ERROR_LIST = []
WARNING_NUM = 0
WARNING_LIST = []
"---------------------------"

if not os.path.exists(TXT_SAVE_PATH):
    os.makedirs(TXT_SAVE_PATH, exist_ok=True)
    
def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args：
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

process_bar = tqdm.tqdm(total=TOTAL_NUM, desc="xml -> yolo", unit='xml')
for xml_name in xml_files:
    process_bar.set_description(f"\033[1;31m{xml_name}\033[0m")
    xml_pre, xml_ext = os.path.splitext(xml_name)  # 分离文件名和后缀
    xml_full_path = os.path.join(XML_PATH, xml_name)  # xml文件完整路径

    # 打开xml文件
    with open(xml_full_path) as fid:
        xml_str = fid.read()
        
    # 将XML字符串编码为字节序列
    xml_bytes = xml_str.encode('utf-8')

    # 使用lxml解析字节序列的XML数据
    xml = etree.fromstring(xml_bytes)
    data = parse_xml_to_dict(xml)["annotation"]
    
    # 构建图片路径
    img_full_path = os.path.join(IMAGE_PATH, xml_pre) + image_type
    
    if os.path.exists(img_full_path):
        img = Image.open(img_full_path)
        img_width, img_height = img.size
        img.close()
    else:  # 图片不存在
        WARNING_NUM += 1
        WARNING_LIST.append(xml_full_path)
        img_width = int(data["size"]["width"])  # 图片宽度
        img_height = int(data["size"]["height"])  # 图片高度
    
    txt_path = os.path.join(TXT_SAVE_PATH, xml_pre + ".txt")
    with open(txt_path, "w") as f:
        # 如果没有object -> 负样本
        objects = data.get("object")
        if objects is None:
            NEG_NUM += 1 
            SUCCEED_NUM += 1
            process_bar.update()
            continue

        for index, obj in enumerate(data["object"]):
            # 获取每个object的box信息
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            class_name = obj["name"]
            
            class_index = class_dict[class_name]

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_full_path))
                ERROR_NUM += 1
                ERROR_LIST.append(xml_name)
                process_bar.update()
                continue
            
            # 检查是否出现越界情况
            if xmin < 0 or ymin < 0 or xmax > img_width or ymax > img_height:
                print("Warning: in '{}' xml, there are out of the boundary".format(xml_full_path))
                ERROR_NUM += 1
                ERROR_LIST.append(xml_name)
                process_bar.update()
                continue

            # 将box信息转换到yolo格式
            xcenter = xmin + (xmax - xmin) / 2  # 中心点的x
            ycenter = ymin + (ymax - ymin) / 2  # 中心点的y
            w = xmax - xmin  # 宽度
            h = ymax - ymin  # 高度

            # 绝对坐标转相对坐标，保存6位小数
            xcenter = round(xcenter / img_width, 6)
            ycenter = round(ycenter / img_height, 6)
            w = round(w / img_width, 6)
            h = round(h / img_height, 6)
            
            # 要输入txt文本的内容
            info = [str(i) for i in [class_index, xcenter, ycenter, w, h]]  # c, x, y, w, h

            # 写入txt
            if index == 0:
                f.write(" ".join(info))
            else:
                f.write("\n" + " ".join(info))
    SUCCEED_NUM += 1
    process_bar.update()
process_bar.close()

print(f"👌 完成："
      f"\n\t成功转换数量 -> {SUCCEED_NUM}/{TOTAL_NUM}"
      f"\n\t创建txt文件的负样本数量 -> {NEG_NUM}"
      f"\n\t出错的数量 -> {ERROR_NUM}"
      f"\n\t警告的数量 -> {WARNING_NUM}")

if ERROR_LIST:
    print('=' * 50)
    for e in ERROR_LIST:
        print(e)
    print(f"🤡 出错了 -> {ERROR_NUM}/{TOTAL_NUM}")
    print('=' * 50)
    
if WARNING_LIST:
    print('=' * 50)
    for warning in WARNING_LIST:
        print(warning)
    print(f"⚠️ 警告: 图片不存在, 使用的是xml中的图片尺寸信息 -> {WARNING_NUM}/{TOTAL_NUM}")
    print('=' * 50)