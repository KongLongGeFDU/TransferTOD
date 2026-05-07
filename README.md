<h2 align="center">TransferTOD: A Generalizable Chinese Multi-Domain Task-Oriented Dialogue System with Transfer Capabilities</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2407.21693"><img src="https://img.shields.io/badge/Paper-Arxiv-blue.svg?style=for-the-badge" alt="Paper"></a>
  <a href="https://aclanthology.org/2024.emnlp-main.710/"><img src="https://img.shields.io/badge/Venue-EMNLP%202024%20Main-orange.svg?style=for-the-badge" alt="EMNLP 2024"></a>
  <a href="https://www.modelscope.cn/models/Mee1ong/TransferTOD-7B"><img src="https://img.shields.io/badge/Model-ModelScope-purple.svg?style=for-the-badge" alt="Model"></a>
</p>

> **Note:** For the Chinese version of this README, please refer to [README_zh.md](README_zh.md).

## 🔔 News

- 🏆 **[2024-09]** Our paper has been accepted at **EMNLP 2024 (Main)**.
- 🤖 **[2024-08]** **TransferTOD-7B** model released on [ModelScope](https://www.modelscope.cn/models/Mee1ong/TransferTOD-7B).
- 🎉 **[2024-07]** Our paper is released on arXiv: [arXiv:2407.21693](https://arxiv.org/abs/2407.21693).

## 📚 Overview

**TransferTOD** is a generalizable Chinese multi-domain Task-Oriented Dialogue (TOD) system with strong **transfer capabilities** to unseen domains. The dataset and the released **TransferTOD-7B** model are designed to handle real-world TOD use cases — slot filling, intent reasoning, and graceful out-of-domain generalization — within a unified framework.

The dataset spans **30 domains** (27 in-domain + 3 held-out OOD: *Water Delivery*, *Sanitation*, *Courier*), and is paired with a two-stage fine-tuning recipe that first injects general TOD ability and then sharpens transfer to specific deployments.

### ✨ Highlights

- 🌐 **30 domains** with **188 slot types** in total — one of the largest publicly available Chinese multi-domain TOD datasets
- 💬 **35,965 turns** across **5,460 dialogues**, with separate **In-Domain** and **Out-of-Domain** test splits
- 🤖 **TransferTOD-7B** model open-sourced on ModelScope, ready for downstream deployment
- 🔁 **Two-stage fine-tuning recipe** balancing general dialogue ability with task-specific transfer
- 🔬 Strong generalization to unseen domains (OOD test on Water Delivery, Sanitation, Courier)

## 📊 Dataset Statistics

<div align="center">

| 📌 **Statistics**     | **Train** | **ID Test** | **OOD Test** |
| --------------------- | --------: | ----------: | -----------: |
| 🌐 # Domains          |        27 |          27 |            3 |
| 🎯 # Slots            |       188 |         188 |           27 |
| 💬 # Dialogues        |     4,320 |         540 |          600 |
| 🔁 # Turns            |    28,680 |       3,585 |        3,700 |
| 📦 # Slots / Dialogue |      10.3 |        10.3 |          9.7 |
| 📏 # Tokens / Turn    |      66.4 |        66.4 |         76.8 |

*Table: Overall statistics of the TransferTOD dataset.*

> **ID Test** = In-Domain test set. **OOD Test** = Out-of-Domain test set, covering three held-out domains: *Water Delivery*, *Sanitation*, and *Courier*.

</div>

## 📂 Project Structure

```
TransferTOD/
├── data/                                   # 📦 All TOD data
│   ├── raw_data/                           # Raw collected data (incl. BELLE 950k)
│   ├── fine_tune_1/                        # Stage-1 fine-tuning data
│   ├── fine_tune_2/                        # Stage-2 fine-tuning data
│   ├── data_generate_template.ipynb        # Data generation template
│   ├── gpt_generate.ipynb                  # GPT-based data generation
│   └── data_process.py                     # Data processing utilities
├── fine_tune/                              # 🛠️ Training scripts
│   ├── fine-tune.py                        # Main training entry
│   ├── ds_config.json                      # DeepSpeed config
│   └── scripts/                            # Full / LoRA fine-tuning launchers
└── inference/                              # 🚀 Inference & evaluation
    ├── inference.py                        # Run inference on the test sets
    ├── eval.py                             # Compute evaluation metrics
    ├── examples.json                       # Example prompts
    └── inference_and_eval.sh               # End-to-end pipeline
```

## 🛠️ Usage Guide

### 1. Prepare Data

All data used in two-stage fine-tuning, along with the raw TransferTOD data, is provided under `data/`. For each stage, `train.json` is a mixture of:

- `train_slot.json` — TOD-specific data, and
- An equivalent amount of `data/raw_data/belle_data/belle_filtered_950k_train.jsonl` — general instruction data.

This balanced mixture preserves general instruction-following ability while injecting strong TOD competence.

### 2. Two-Stage Fine-tuning

**Full fine-tuning:**

```bash
bash fine_tune/scripts/finetune_full.sh
```

**LoRA fine-tuning:**

```bash
bash fine_tune/scripts/finetune_lora.sh
```

Adjust `model_name_or_path`, `data_path`, and DeepSpeed settings in `ds_config.json` before launching.

### 3. Inference & Evaluation

End-to-end inference + evaluation on the TransferTOD test set:

```bash
bash inference/inference_and_eval.sh
```

This will:

1. Run `inference.py` on the **ID** and **OOD** test sets.
2. Run `eval.py` to compute slot-level and dialogue-level metrics.

### 4. Use the Released Model

The fine-tuned **TransferTOD-7B** is available on ModelScope:

🤖 **[Mee1ong/TransferTOD-7B](https://www.modelscope.cn/models/Mee1ong/TransferTOD-7B)**

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Mee1ong/TransferTOD-7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Mee1ong/TransferTOD-7B", trust_remote_code=True)
```

## 📝 Citation

If you find this project useful in your research, please cite us:

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
    doi       = "10.18653/v1/2024.emnlp-main.710",
    pages     = "12750--12771"
}
```

## 🔗 Related Projects

| Project | Description | Link |
|---------|-------------|------|
| **PFDial** (ACL 2025) | Structured dialogue instruction tuning based on UML flowcharts | [GitHub](https://github.com/KongLongGeFDU/PFDial) |
| **LLMEval-Med** (EMNLP 2025) | Real-world clinical benchmark for medical LLMs | [GitHub](https://github.com/llmeval/LLMEval-Med) |
| **LLMEval-Fair** (ACL 2026) | Robust & fair evaluation, 200K+ questions | [GitHub](https://github.com/llmeval/LLMEval-Fair) |

## 📞 Contact Us

For questions or collaboration, please:

- Open an [Issue](https://github.com/KongLongGeFDU/TransferTOD/issues) on GitHub
- Contact the project maintainers:
  - **Ming Zhang**: mingzhang23@m.fudan.edu.cn

---

<p align="center">
  <b>TransferTOD</b> | Fudan NLP Lab
</p>
