import json
import random
import sys
import re
import os
from typing import List, Dict
from utils import *

ROW_DATA_PATH = "./data/raw_data"  # 暂时只有ood部分用到，待定

SYS_PROMPT = "你是一个负责信息抽取的机器人，需要信息抽取的场景是“{domain}”。请你根据与用户的对话填充槽位，并不断对槽位为空的部分进行提问，每一次提问的槽位数量为{extract_slot}。如果用户所回答的内容中，有不属于你上轮对话中提问的槽位，那么请不要将用户回答内容中出错的部分填入槽位，而是对用户回答中出错的槽位进行重新询问。"

SEED = 42
OOD_DATA_INDEX = [28, 29, 30]  # 作为ood data 的原始数据的domain索引
MIX_RATIO = 1  # belle数据和slot数据数据混合比例
MIX_LEN = None

assert (
    MIX_RATIO is not None or MIX_LEN is not None
), " len_belle_data and mix_ratio cannot be both None at the same time"


# ========================================================================下述是处理数据处理相关========================================================================


class BaseDataProcess_ID:
    def __init__(
        self,
        belle_data_path,
        data_path_list,
        ood_index_list,
        output_data_path,
        slot_data_split_ratio: List[int] = [8, 1, 1],
        data_mix_ratio=1,
        save_train_slot_data=False,  # 是否单独保存train_slot的数据
    ) -> None:
        self.data_path_list = data_path_list
        self.belle_data_path = belle_data_path
        self.ood_index_list = ood_index_list
        self.output_data_path = output_data_path

        # 处理slot数据
        slot_data_train, slot_data_test, slot_data_valid = self.process_slot_data(
            data_path_list=data_path_list,
            ood_index_list=ood_index_list,
            slot_data_split_ratio=slot_data_split_ratio,
        )
        self.len_slot_data_train = len(slot_data_train)
        self.len_slot_data_test = len(slot_data_test)
        if slot_data_valid:
            self.len_slot_data_valid = len(slot_data_valid)

        # 处理belle数据
        len_belle = int(self.len_slot_data_train * data_mix_ratio)
        belle_data = self.process_belle_data(
            belle_data_path=belle_data_path, len_belle=len_belle
        )
        self.len_belle_data = len(belle_data)

        # 合并slot_train数据和belle数据
        train_data = belle_data + slot_data_train
        test_data = slot_data_test
        valid_data = slot_data_valid

        # 输出数据json文件
        train_data_path = os.path.join(output_data_path, "train.json")
        data2json(train_data, train_data_path)
        test_data_path = os.path.join(output_data_path, "test.json")
        data2json(test_data, test_data_path)
        if slot_data_valid:
            valid_data_path = os.path.join(output_data_path, "valid.json")
            data2json(valid_data, valid_data_path)
        if save_train_slot_data:
            train_slot_data_path = os.path.join(output_data_path, "train_slot.json")
            data2json(slot_data_train, train_slot_data_path)

        # 输出数据集基本信息
        self.output_information(test_data=test_data)

    def _load_data(self, data_path: str, ood_index_list: List[str]) -> List[List]:
        # filter ood folder_path
        folder_names = get_folder_names(data_path)
        filter_folder_names = []
        for folder_name in folder_names:
            flag = True
            for ood_index in ood_index_list:
                if str(ood_index) in folder_name:
                    flag = False
                    break
            if flag:
                filter_folder_names.append(folder_name)
        filter_folder_paths = [
            os.path.join(data_path, filter_folder_name)
            for filter_folder_name in filter_folder_names
        ]

        # load json data
        data_list = []
        for filter_folder_path in filter_folder_paths:
            domain_data_files = get_file_names(filter_folder_path)
            domain_data_files_path = [
                os.path.join(filter_folder_path, domain_data_file)
                for domain_data_file in domain_data_files
            ]
            domain_data_list = [
                load_json(domain_data_file_path)
                for domain_data_file_path in domain_data_files_path
            ]
            data_list.append(domain_data_list)

        return data_list

    def _fill_sysprompt(self, sysprompt, extract_slot, domain, null_num):
        # 需要单独考虑倒数第二轮问题数目和规定询问的slot数目不一样的情况
        if null_num > 0 and null_num < extract_slot:
            return sysprompt.format(domain=domain, extract_slot=null_num)
        else:
            return sysprompt.format(domain=domain, extract_slot=extract_slot)

    def _sub_format_data_slot(self, data: List):
        """
        训练数据格式
        {
            "conversations":[
                {
                    "from": "human"
                    "value": "<sysprompt>\n
                        槽位：{\"姓名\":null,\"时间\":null,\"地点\":null}\n
                        上轮对话：{\"gpt\": \"\", \"human\": \"\"}"
                },
                {
                    "from": "gpt"
                    "value": "{\"姓名\":null,\"时间\":null,\"地点\":null} 您好，请  问您叫什么呢？"
                }
            ]
        }
        """

        format_data = []

        for da in data:
            extract_slot = da["extract_slot"]  # 抽slot数目
            domain = da["task"]  # 场景
            for d in da["content"]:
                # 处理sysprompt（注意倒数第二轮可能问的问题数目和规定的数目不一样，单独处理）
                null_num = sum(value is None for value in d["new_slots"].values())
                sysprompt_fill = self._fill_sysprompt(
                    SYS_PROMPT, extract_slot, domain, null_num
                )

                conv = {}
                conv["conversations"] = []
                if d["conversations"][0]["value"] != "":
                    origin_slots = json.dumps(d["origin_slots"], ensure_ascii=False)
                    human_answer = d["conversations"][0]["value"]
                    conv["conversations"].append(
                        {
                            "from": "human",
                            "value": f'{sysprompt_fill}\n槽位：{origin_slots}\n上轮对话：{{"gpt"："{question}","human"："{human_answer}"}}',
                        }
                    )
                else:
                    conv["conversations"].append(
                        {
                            "from": "human",
                            "value": sysprompt_fill
                            + "\n槽位："
                            + json.dumps(d["origin_slots"], ensure_ascii=False)
                            + '\n上轮对话：{"gpt"："","human"：""}',
                        }
                    )
                question = d["conversations"][1]["value"]
                try:
                    conv["conversations"].append(
                        {
                            "from": "gpt",
                            "value": json.dumps(d["new_slots"], ensure_ascii=False)
                            + " "
                            + question,
                        }
                    )
                except:
                    print(json.dumps(d["new_slots"], ensure_ascii=False))
                    print(question)
                format_data.append(conv)

        return format_data

    def _format_data_slot(self, data: List[List]) -> List[List]:
        format_data = []
        for domain_data in data:
            format_data_domain = [self._sub_format_data_slot(dd) for dd in domain_data]
            format_data.append(format_data_domain)
        return format_data

    def _split_json_data(self, data, split_ratio: List[int] = [8, 1, 1], shuffle=False):
        """
        划分训练集、测试集和验证集，默认划分比例811
        """

        # 计算划分的索引
        sum_ratios = sum(split_ratio)
        split_ratios = [ratio / sum_ratios for ratio in split_ratio]

        total_samples = len(data)
        train_samples = int(total_samples * split_ratios[0])
        test_samples = int(total_samples * split_ratios[1])
        # 随机打乱数据顺序
        if shuffle:
            # 设置随机数种子
            random.seed(SEED)
            random.shuffle(data)
        # 划分为训练集和测试集
        train_data = data[:train_samples]
        test_data = data[train_samples : train_samples + test_samples]
        valid_data = (
            data[train_samples + test_samples :] if split_ratios[2] != 0 else []
        )

        return train_data, test_data, valid_data

    def _split_data(
        self, format_data: List[List], slot_data_split_ratio: List[int] = [8, 1, 1]
    ):
        format_data_train = []  # each data(eg:clean) a data_train
        format_data_test = {}  # each data(eg:clean) a data_test
        format_data_valid = []  # each data(eg:clean) a data_valid

        for format_data_domain in format_data:
            format_data_domain_train = []
            format_data_domain_test = []
            format_data_domain_valid = []

            for fdd in format_data_domain:
                train_data, test_data, valid_data = self._split_json_data(
                    fdd, slot_data_split_ratio
                )
                format_data_domain_train.extend(train_data)
                format_data_domain_test.extend(test_data)
                if valid_data:
                    format_data_domain_valid.extend(valid_data)

            # get domain name for test
            domain_name = format_data_domain[0][0]["conversations"][0]["value"]
            match = re.search(r"需要信息抽取的场景是“(.*?)”。", domain_name)
            if match:
                domain_name = match.group(1)
            else:
                print("未找到匹配到domain_name")
            # extend test data
            format_data_test.setdefault(domain_name, []).extend(format_data_domain_test)
            # extend train data
            format_data_train.extend(format_data_domain_train)
            # extend valid data
            if format_data_domain_valid:
                format_data_valid.extend(format_data_domain_valid)

        return format_data_train, format_data_test, format_data_valid

    def process_slot_data(
        self,
        data_path_list: List[str],
        ood_index_list: List[str],
        slot_data_split_ratio: List[int] = [8, 1, 1],
    ):
        # get data name
        data_names = [dp[dp.rfind("/") + 1 :] for dp in data_path_list]

        # load data (filter ood data)
        datas = [
            self._load_data(data_path, ood_index_list) for data_path in data_path_list
        ]

        # format data
        format_datas = [self._format_data_slot(data) for data in datas]

        # split data
        split_datas_train = []
        split_datas_test = {}
        split_datas_valid = []
        for format_data, data_name in zip(format_datas, data_names):
            train_data, test_data, valid_data = self._split_data(
                format_data, slot_data_split_ratio
            )
            split_datas_train.append(train_data)
            split_datas_test.update({data_name: test_data})
            if valid_data:
                split_datas_valid.append(valid_data)

        # merge train datas (eg: merge clean and noise)
        merge_train_data = []
        for split_data_train in split_datas_train:
            merge_train_data.extend(split_data_train)

        # if split_datas_valid is not None, merge data
        merge_valid_data = []
        if split_datas_valid:
            for split_data_valid in split_datas_valid:
                merge_valid_data.extend(split_data_valid)

        # shuffle train data
        random.seed(SEED)
        random.shuffle(merge_train_data)

        # shuffle valid data
        if split_datas_valid:
            random.shuffle(merge_valid_data)

        return merge_train_data, split_datas_test, merge_valid_data

    def _format_data_belle(self, samples: List):
        data = []
        for d in samples:
            conv = {}
            conv["conversations"] = []
            conv["conversations"].append(
                {"from": "human", "value": d["instruction"] + d["input"]}
            )
            conv["conversations"].append({"from": "gpt", "value": d["output"]})
            data.append(conv)
        return data

    def process_belle_data(self, belle_data_path: str, len_belle: int):
        # 设置随机数种子
        random.seed(SEED)
        train = []
        with open(belle_data_path) as f:
            for line in f:
                data = json.loads(line)
                train.append(data)

        train_sample = random.sample(train, len_belle)
        data = self._format_data_belle(train_sample)
        return data

    def output_information(self, test_data: Dict):
        test_info = {}
        for data_name, values in test_data.items():
            data_info = {k: len(v) for k, v in values.items()}
            data_sum = sum([len(v) for v in values.values()])
            test_info.setdefault(
                data_name, {"details": data_info, "data_sum": data_sum}
            )
        test_info.setdefault(
            "data_sum", sum([value["data_sum"] for value in test_info.values()])
        )

        info = {
            "len_belle_data": self.len_belle_data,
            "len_slot_data_train": self.len_slot_data_train,
            "len_slot_data_valid": (
                self.len_slot_data_valid if self.len_slot_data_valid else 0
            ),
            "len_slot_data_test": test_info,
        }
        info_data_path = os.path.join(self.output_data_path, "info.json")
        data2json(info, info_data_path)


class BaseDataProcess_OOD:
    pass


# ========================================================================下述是处理 fine_tune_1 数据相关========================================================================


def get_1_clean_data():
    """处理生成ID数据：10%clean"""
    data_path_list = ["./data/raw_data/clean"]
    ood_index_list = OOD_DATA_INDEX
    output_data_path = "./data/fine_tune_1/v1"
    belle_data_path = (
        "./data/raw_data/belle_data/belle_filtered_950k_train.jsonl"
    )
    BaseDataProcess_ID(
        belle_data_path=belle_data_path,
        data_path_list=data_path_list,
        ood_index_list=ood_index_list,
        output_data_path=output_data_path,
        save_train_slot_data=True,
    )


def get_2_clean_noise_data():
    """处理生成ID数据：10%clean + noise"""
    data_path_list = ["./data/raw_data/clean", "./data/raw_data/noise"]
    ood_index_list = OOD_DATA_INDEX
    output_data_path = "./data/fine_tune_1/v2"
    belle_data_path = (
        "./data/raw_data/belle_data/belle_filtered_950k_train.jsonl"
    )
    BaseDataProcess_ID(
        belle_data_path=belle_data_path,
        data_path_list=data_path_list,
        ood_index_list=ood_index_list,
        output_data_path=output_data_path,
        save_train_slot_data=True,
    )


def get_3_clean_noise_gpt_data():
    """处理生成ID数据：10%clean_gpt + noise_gpt"""
    data_path_list = ["./data/raw_data/clean_gpt", "./data/raw_data/noise_gpt"]
    ood_index_list = OOD_DATA_INDEX
    output_data_path = './data/fine_tune_1/v3'
    belle_data_path = (
        "./data/raw_data/belle_data/belle_filtered_950k_train.jsonl"
    )
    BaseDataProcess_ID(
        belle_data_path=belle_data_path,
        data_path_list=data_path_list,
        ood_index_list=ood_index_list,
        output_data_path=output_data_path,
        save_train_slot_data=True,
    )



def get_4_clean_noise_gpt_human_data():
    """处理生成ID数据：10%clean_gpt_human + noise_gpt_human"""
    data_path_list = ["./data/raw_data/clean_gpt_human", "./data/raw_data/noise_gpt_human"]
    ood_index_list = OOD_DATA_INDEX
    output_data_path = './data/fine_tune_1/v4'
    belle_data_path = (
        "./data/raw_data/belle_data/belle_filtered_950k_train.jsonl"
    )
    BaseDataProcess_ID(
        belle_data_path=belle_data_path,
        data_path_list=data_path_list,
        ood_index_list=ood_index_list,
        output_data_path=output_data_path,
        save_train_slot_data=True,
    )



get_1_clean_data()
get_2_clean_noise_data()
get_3_clean_noise_gpt_data()
get_4_clean_noise_gpt_human_data()


# ========================================================================下述是处理 fine_tune_2 数据相关========================================================================


# ========================================================================下述是处理 OOD 数据相关========================================================================
# 选择最离群的三个domain作为ood
def choice_ood():
    data_path = f"{ROW_DATA_PATH}/slot_data"
    folder_names = get_folder_names(data_path)

    slots_list = []

    for folder_name in folder_names:
        domain_path = f"{data_path}/{folder_name}"
        file_names = get_file_names(domain_path)
        first_file_path = f"{domain_path}/{file_names[0]}"
        with open(first_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        task = folder_name
        slots = list(data[0]["content"][0]["origin_slots"].keys())
        slots_list.append({task: slots})

    count = {}
    for i in range(len(slots_list)):
        check_task = list(slots_list[i].keys())[0]
        check_slots = list(slots_list[i].values())[0]
        check_slots_set = set(check_slots)

        merged_set = set()
        for j in range(len(slots_list)):
            if i == j:
                continue
            merge_list = list(slots_list[j].values())[0]
            merged_set.update(merge_list)

        count_check = check_slots_set & merged_set
        count.update({check_task: len(count_check)})

    sorted_count = dict(sorted(count.items(), key=lambda item: item[1]))

    # 打印slot相交数目
    for item in sorted_count.items():
        print(item)
    """打印内容如下,选取ood:30、29、28
        ('30-courier', 0)
        ('26-print', 2)
        ('27-photography', 3)
        ('29-sanitation', 3)
        ('14-shopping', 3)
        ('20-gym', 3)
        ('15-calling_card', 3)
        ('8-car_rent', 3)
        ('28-water_delivery', 3)
        ('1-hotel', 3)
        ('11-concert', 4)
        ('22-wash', 4)
        ('7-ocean_liner', 4)
        ('2-tourism', 4)
        ('9-house_rent', 4)
        ('25-solicitors', 4)
        ('4-takeaways', 5)
        ('3-gastronomy', 5)
        ('18-beautysalon', 5)
        ('24-removal', 5)
        ('10-movie', 5)
        ('6-air_ticket', 5)
        ('17-dentistry', 5)
        ('21-training_session', 5)
        ('16-hospital', 5)
        ('12-museum', 5)
        ('13-vacation_villages', 5)
        ('5-train_ticket', 5)
        ('23-appliance_repair', 6)
        ('19-hairsalon', 7)
    """

    # 看看去重slot总数
    sum_slot_clean = set()
    for j in range(len(slots_list)):
        sum_list = list(slots_list[j].values())[0]
        sum_slot_clean.update(sum_list)
    print(sum_slot_clean)
    print(f"去重slot总数：{len(sum_slot_clean)}")  # 去重slot总数205


def get_ood_data(ood_data_index=[]):
    data_path = f"{ROW_DATA_PATH}/slot_data"
    all_folder_names = get_folder_names(data_path)

    ood_data = {}

    for folder_name in all_folder_names:
        for index in ood_data_index:
            if str(index) in folder_name:
                domain_data = []
                # process ood data
                ood_folder_path = f"{data_path}/{folder_name}"
                ood_file_names = get_file_names(ood_folder_path)
                for ood_file_name in ood_file_names:
                    file_path = f"{ood_folder_path}/{ood_file_name}"
                    with open(file_path, "r", encoding="utf-8") as file:
                        data = json.load(file)
                    data = format_data(data)
                    domain_data += data
                ood_data.update({folder_name: domain_data})

    return ood_data
