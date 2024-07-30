import json
import os
import pandas as pd


# ========================下述是处理json相关========================


def load_json(json_file_path: str):
    """加载json文件数据

    Args:
        json_file_path (str): json文件路径

    Returns:
        Any: json数据
    """
    try:
        # 打开并读取JSON文件
        with open(json_file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            return data

    except FileNotFoundError:
        print(f"未找到文件：{json_file_path}")
    except json.JSONDecodeError:
        print(f"JSON文件解析错误: {json_file_path}")


def data2json(data, json_file: str):
    """data数据输出成json文件

    Args:
        data (Any): 输出数据
        json_file (str): 保存json文件的路径
    """
    # 检查路径是否存在，如果不存在则创建它
    directory = os.path.dirname(json_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 使用with语句打开文件，将数    据写入文件中
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def jsonl2json(jsonl_file: str, json_file: str):
    """jsonl文件转成json文件

    Args:
        jsonl_file (str): jsonl文件路径
        json_file (str): json文件路径
    """
    # 打开输入文件和输出文件
    with open(jsonl_file, "r", encoding="utf-8") as input_file:
        # 逐行读取 JSONL 文件，并将每行的 JSON 对象解析为 Python 字典
        data = [json.loads(line) for line in input_file]
        # 将每个 JSON 对象写入输出文件
    data2json(data, json_file)


def parquet2json(parquet_file: str, output_path: str):
    """parquet转json

    Args:
        parquet_file (str): parquet文件路径
        output_path (str): 文件输出路径(非文件名)
    """
    # 读取Parquet文件
    df = pd.read_parquet(parquet_file)

    # 获取文件名
    output_file_name = parquet_file[parquet_file.rfind("/") + 1 :]
    output_file_name = output_file_name[: output_file_name.find("-")]
    output_file = f"{output_path}/{output_file_name}.json"

    # # 输出结果到json文件
    df.to_json(output_file, orient="records")


# ========================下述是处理文件相关========================


def get_folder_names(query_path: str):
    """获取路径下所有folder名称

    Args:
        query_path (str): 查询路径

    Returns:
        list: 返回查询列表
    """
    return [
        folder
        for folder in os.listdir(query_path)
        if os.path.isdir(os.path.join(query_path, folder))
    ]


def get_file_names(folder_path: str):
    """获取路径下所有文件名

    Args:
        folder_path (str): 查询的文件夹路径

    Returns:
        _type_: 返回查询列表
    """
    return [
        file
        for file in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file))
    ]
