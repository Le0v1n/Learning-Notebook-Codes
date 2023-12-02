"""
+ 脚本说明：用于将xml格式标注文件转换为json格式
+ 用途：labelImg -> labelme
+ 要求：要有对应的图片文件，这样读取到的尺寸信息是最准确的。
"""
import os
import tqdm
from lxml import etree
from PIL import Image
import json


"""============================ 需要修改的地方 ==================================="""
IMAGE_PATH = "EXAMPLE_FOLDER/images"  # 原图文件夹路径
XML_PATH = "EXAMPLE_FOLDER/annotations-xml"  # 保存xml文件夹路径
JSON_SAVE_PATH = "EXAMPLE_FOLDER/annotations-json"

IMAGE_TYPE = '.jpg'
OVERRIDE = True  # 是否覆盖已存在的json文件

classes_dict = {
    '0': "cat",
    '1': 'dog'
}
"""==============================================================================="""

assert os.path.exists(IMAGE_PATH), f"图像文件夹[{IMAGE_PATH}]不存在!"
assert os.path.exists(XML_PATH), f"xml文件夹[{XML_PATH}]不存在!"

os.makedirs(JSON_SAVE_PATH) if not os.path.exists(JSON_SAVE_PATH) else None

xml_file_list = [file for file in os.listdir(XML_PATH) if file.endswith(".xml")]

"------------计数------------"
TOTAL_NUM = len(xml_file_list)
SUCCEED_NUM = 0  # 成功创建xml数量
SKIP_NUM = 0  # 跳过创建xml文件数量
OBJECT_NUM = 0  # object数量
WARNING_NUM = 0  # 没有对应图片
WARNING_LIST = []
"---------------------------"


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

process_bar = tqdm.tqdm(total=TOTAL_NUM, desc="xml2json", unit='.xml')
for i, xml_name in enumerate(xml_file_list):
    process_bar.set_description(f"Process in \033[1;31m{xml_name}\033[0m")
    xml_pre, xml_ext = os.path.splitext(xml_name)  # 分离前缀和后缀
    
    xml_full_path = os.path.join(XML_PATH, xml_name)  # xml文件完整路径
    json_save_path = os.path.join(JSON_SAVE_PATH, xml_pre) + '.json'  # xml文件完整路径

    if not OVERRIDE and os.path.exists(json_save_path):  # 目标json文件存在 -> 跳过
        SKIP_NUM += 1
        process_bar.update()
        continue

    # 打开xml文件
    with open(xml_full_path) as fid:
        xml_str = fid.read()
        
    # 将XML字符串编码为字节序列
    xml_bytes = xml_str.encode('utf-8')

    # 使用lxml解析字节序列的XML数据
    xml = etree.fromstring(xml_bytes)
    data = parse_xml_to_dict(xml)["annotation"]
    
    # 构建图片路径
    img_full_path = os.path.join(IMAGE_PATH, xml_pre) + IMAGE_TYPE
    
    # 图片存在 -> 获取图片的宽度和高度(确保这两个值是正确的)
    if os.path.exists(img_full_path):
        img = Image.open(img_full_path)
        img_width, img_height = img.size
        img.close()
    else:  # 图片不存在 -> 获取 xml 中的图片高度和宽度
        WARNING_NUM += 1
        WARNING_LIST.append(xml_full_path)
        img_width = int(data["size"]["width"])  # 图片宽度
        img_height = int(data["size"]["height"])  # 图片高度

    # 创建要保存的json数据字典
    json_data = {
        "version": "0.2.2",
        "flags": {},
        "shapes": [],
        "imagePath": f"{xml_pre}{IMAGE_TYPE}",
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width,
        "text": ""
    }

    # 处理每个 object
    for obj in data.get("object", []):
        label = obj["name"]
        xmin = float(obj["bndbox"]["xmin"])
        ymin = float(obj["bndbox"]["ymin"])
        xmax = float(obj["bndbox"]["xmax"])
        ymax = float(obj["bndbox"]["ymax"])

        # 计算中心点坐标和宽高
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin

        # 添加到 shapes 列表中
        json_data["shapes"].append({
            "label": label,
            "text": "",
            "points": [
                [xmin, ymin],
                [xmax, ymax]
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        })
        OBJECT_NUM += 1

    # 保存为json文件
    with open(json_save_path, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=2)
    
    SUCCEED_NUM += 1
    process_bar.update()
process_bar.close()

for _warning in WARNING_LIST:
    print(f"⚠️ {_warning}")

print(f"👌 xml2json已完成, 详情如下:"
      f"\n\t成功转换文件数量/总文件数量 = \033[1;32m{SUCCEED_NUM}\033[0m/{TOTAL_NUM}"
      f"\n\t跳过转换文件数量/总文件数量 = \033[1;31m{SKIP_NUM}\033[0m/{TOTAL_NUM}"
      f"\n\t所有样本的 object 数量/总文件数量 = \033[1;32m{OBJECT_NUM}\033[0m/{TOTAL_NUM}"
      f"\n\t平均每个json文件中object的数量为: {int(OBJECT_NUM / SUCCEED_NUM)}"
      f"\n\t ⚠️没有对应图片的数量为: {WARNING_NUM}"
      f"\n\n\t结果保存路径为: \033[1;31m{JSON_SAVE_PATH}\033[0m")

if SUCCEED_NUM + SKIP_NUM == TOTAL_NUM:
    print(f"\n👌 \033[1;32mNo Problem\033[0m")
else:
    print(f"\n🤡 \033[1;31m貌似有点问题, 请仔细核查!\033[0m")