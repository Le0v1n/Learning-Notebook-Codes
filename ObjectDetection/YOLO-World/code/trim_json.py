from pathlib import Path
import json

def trim_data(data, max_items):
    if isinstance(data, dict):
        # 如果是字典，遍历字典并递归调用trim_data
        return {key: trim_data(value, max_items) for key, value in data.items()}
    elif isinstance(data, list):
        # 如果是列表，只保留前max_items个元素
        return [trim_data(item, max_items) for item in data[:max_items]]
    else:
        # 基础数据类型，直接返回
        return data

def read_and_trim_json(json_path: str, json_save_path: str = None, max_items: int = 10) -> str:
    json_path = Path(json_path)
    json_save_path = Path(json_save_path) if json_save_path else json_path.parent.joinpath(f"{json_path.stem}-trim.json")
    
    try:
        with json_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 递归地截取数据
        trimmed_data = trim_data(data, max_items)
        
        # 将处理后的数据写入新的JSON文件
        with json_save_path.open('w', encoding='utf-8') as f:
            json.dump(trimmed_data, f, ensure_ascii=False, indent=4)
        
        return str(json_save_path)

    except FileNotFoundError:
        print(f"❌ 文件 {str(json_path)} 未找到。")
    except json.JSONDecodeError:
        print(f"❌ 文件 {str(json_path)} 不是有效的JSON格式。")
    except Exception as e:
        print(f"❌ 处理 {str(json_path)} 时发生错误：{e}")

if __name__ == "__main__":
    # 使用示例
    input_json_path = 'ObjectDetection/YOLO-World/code/cc3m_pseudo_annotations.json'
    output_path = read_and_trim_json(
        input_json_path
    )
    print(f"处理完成，文件已保存到：{output_path}")