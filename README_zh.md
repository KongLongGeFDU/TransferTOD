<h2 align="center">TransferTOD：可迁移的中文多领域任务型对话系统</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2407.21693"><img src="https://img.shields.io/badge/论文-Arxiv-blue.svg?style=for-the-badge" alt="论文"></a>
  <a href="https://aclanthology.org/2024.emnlp-main.710/"><img src="https://img.shields.io/badge/会议-EMNLP%202024%20Main-orange.svg?style=for-the-badge" alt="EMNLP 2024"></a>
  <a href="https://www.modelscope.cn/models/Mee1ong/TransferTOD-7B"><img src="https://img.shields.io/badge/模型-ModelScope-purple.svg?style=for-the-badge" alt="模型"></a>
</p>

> **注意：** 英文版 README 请参阅 [README.md](README.md)。

## 🔔 最新消息

- 🏆 **[2024-09]** 论文被 **EMNLP 2024 主会** 录用。
- 🤖 **[2024-08]** **TransferTOD-7B** 模型发布于 [ModelScope](https://www.modelscope.cn/models/Mee1ong/TransferTOD-7B)。
- 🎉 **[2024-07]** 论文发布于 arXiv：[arXiv:2407.21693](https://arxiv.org/abs/2407.21693)。

## 📚 项目简介

**TransferTOD** 是一个具有强**迁移能力**、可泛化到未见领域的中文多领域任务型对话（TOD）系统。本项目同时开放了配套数据集与微调后的 **TransferTOD-7B** 模型，能够在统一框架下完成槽位填充、意图推理以及优雅的领域外（OOD）泛化。

数据集共覆盖 **30 个领域**（27 个域内 + 3 个保留外域：*送水*、*环卫*、*快递*），并配套了一套**两阶段微调**方案：第一阶段注入通用 TOD 能力，第二阶段强化对具体部署场景的迁移。

### ✨ 核心亮点

- 🌐 **30 个领域**、**188 个槽位类型**——目前最大规模的公开中文多领域 TOD 数据集之一
- 💬 **5,460** 段对话、**35,965** 轮交互，并提供独立的**域内**与**域外**测试集
- 🤖 **TransferTOD-7B** 模型已开源至 ModelScope，开箱即用
- 🔁 **两阶段微调方案**，兼顾通用对话能力与任务迁移能力
- 🔬 在三个保留域（送水、环卫、快递）上展现出强泛化能力

## 📊 数据统计

<div align="center">

| 📌 **统计项**         | **训练集** | **ID 测试集** | **OOD 测试集** |
| --------------------- | ---------: | ------------: | -------------: |
| 🌐 领域数量            |         27 |            27 |              3 |
| 🎯 槽位数量            |        188 |           188 |             27 |
| 💬 对话数量            |      4,320 |           540 |            600 |
| 🔁 总轮次              |     28,680 |         3,585 |          3,700 |
| 📦 平均槽位/对话       |       10.3 |          10.3 |            9.7 |
| 📏 平均 Token/轮次     |       66.4 |          66.4 |           76.8 |

*表：TransferTOD 数据集整体统计数据。*

> **ID Test** 表示域内（In-Domain）测试集；**OOD Test** 表示域外（Out-of-Domain）测试集，覆盖三个保留外域：*送水*、*环卫*、*快递*。

</div>

## 📂 项目结构

```
TransferTOD/
├── data/                                   # 📦 全部 TOD 数据
│   ├── raw_data/                           # 原始采集数据（含 BELLE 950k）
│   ├── fine_tune_1/                        # 第一阶段微调数据
│   ├── fine_tune_2/                        # 第二阶段微调数据
│   ├── data_generate_template.ipynb        # 数据生成模板
│   ├── gpt_generate.ipynb                  # 基于 GPT 的数据生成
│   └── data_process.py                     # 数据处理工具
├── fine_tune/                              # 🛠️ 训练脚本
│   ├── fine-tune.py                        # 训练入口
│   ├── ds_config.json                      # DeepSpeed 配置
│   └── scripts/                            # 全参 / LoRA 微调启动脚本
└── inference/                              # 🚀 推理与评测
    ├── inference.py                        # 在测试集上推理
    ├── eval.py                             # 计算评测指标
    ├── examples.json                       # 示例 prompt
    └── inference_and_eval.sh               # 端到端流程
```

## 🛠️ 使用指南

### 1. 数据准备

两阶段微调所用的全部数据以及 TransferTOD 原始数据均位于 `data/`。每个阶段的 `train.json` 由以下两部分等量混合得到：

- `train_slot.json`——TOD 任务相关数据
- `data/raw_data/belle_data/belle_filtered_950k_train.jsonl` 中等量样本——通用指令数据

这种均衡混合可在保留通用指令遵循能力的同时，注入强 TOD 能力。

### 2. 两阶段微调

**全参微调：**

```bash
bash fine_tune/scripts/finetune_full.sh
```

**LoRA 微调：**

```bash
bash fine_tune/scripts/finetune_lora.sh
```

请在启动前修改 `model_name_or_path`、`data_path` 以及 `ds_config.json` 中的 DeepSpeed 配置。

### 3. 推理与评测

在 TransferTOD 测试集上的端到端推理 + 评测：

```bash
bash inference/inference_and_eval.sh
```

该脚本将依次：

1. 通过 `inference.py` 在 **ID** 与 **OOD** 测试集上推理。
2. 通过 `eval.py` 计算槽位级与对话级评测指标。

### 4. 使用已发布模型

微调后的 **TransferTOD-7B** 已发布在 ModelScope：

🤖 **[Mee1ong/TransferTOD-7B](https://www.modelscope.cn/models/Mee1ong/TransferTOD-7B)**

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Mee1ong/TransferTOD-7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Mee1ong/TransferTOD-7B", trust_remote_code=True)
```

## 📝 引用

如果本项目对您的研究有帮助，欢迎引用：

```bibtex
@inproceedings{zhang-etal-2024-transfertod,
    title     = "{T}ransfer{TOD}: A Generalizable {C}hinese Multi-Domain Task-Oriented
                 Dialogue System with Transfer Capabilities",
    author    = "Zhang, Ming and Huang, Caishuang and Wu, Yilong and Liu, Shichun and
                 Zheng, Huiyuan and Dong, Yurui and Shen, Yujiong and Dou, Shihan and
                 Zhao, Jun and Ye, Junjie and Zhang, Qi and Gui, Tao and Huang, Xuanjing",
    editor    = "Al-Onaizan, Yaser and Bansal, Mohit and Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural
                 Language Processing",
    month     = nov,
    year      = "2024",
    address   = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2024.emnlp-main.710/",
    pages     = "12750--12771"
}
```

## 🔗 相关项目

| 项目 | 说明 | 链接 |
|------|------|------|
| **PFDial**（ACL 2025） | 基于 UML 流程图的结构化对话指令微调 | [GitHub](https://github.com/KongLongGeFDU/PFDial) |
| **LLMEval-Med**（EMNLP 2025） | 真实临床场景下的医学大模型基准 | [GitHub](https://github.com/llmeval/LLMEval-Med) |
| **LLMEval-Fair**（ACL 2026） | 鲁棒公平的大模型评测，20 万+题 | [GitHub](https://github.com/llmeval/LLMEval-Fair) |

## 📞 联系我们

如有问题或合作意向，请：

- 在 GitHub 上提交 [Issue](https://github.com/KongLongGeFDU/TransferTOD/issues)
- 联系项目维护者：
  - **张明（Ming Zhang）**：mingzhang23@m.fudan.edu.cn

---

<p align="center">
  <b>TransferTOD</b> | 复旦大学自然语言处理实验室
</p>
