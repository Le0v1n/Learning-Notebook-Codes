from pathlib import Path
import json


def read_json(jsonfile: Path) -> dict:
    with jsonfile.open('r') as f:
        return json.load(f)
    
    
if __name__ == "__main__":
    json_path = Path('ObjectDetection/YOLO-World/code/instances_val2017.json')
    json_dict = read_json(json_path)
    
    # print(f"{json_dict.keys()}")
    # exit()
    
    for key in json_dict.keys():
        value = json_dict[key]
        
        if isinstance(value, list):
            print(f"{key}: {value[:40]}")
        else:
            print(f"{key}: {value}")
        print()
        
    
    # print(f"{lvis_dict.keys()}")  # ['categories', 'info', 'licenses', 'images', 'annotations']
    # print(len(lvis_dict['categories']))
    # print(lvis_dict['categories'][0])