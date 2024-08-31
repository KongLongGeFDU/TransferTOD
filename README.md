# TransferTOD
The code repository of paper "TransferTOD: A Generalizable Chinese Multi-Domain Task-Oriented Dialogue System with Transfer Capabilities"

> All the data used in two-staged finetuning and the raw data of TransferTOD is included in directory `./data`. For each version, `train.json` is a mixed data of `train_slot.json` and equivalent amounts of `./data/raw_data/belle_data/belle_filtered_950k_train.jsonl`
>
> For full fine-tuning, run `./fine_tune/scripts/finetune_full.sh`, while for lora fine-tuning, run `./fine_tune/scripts/finetune_lora.sh`.
>
> For inference and evaluation with the TransferTOD test set, run `./inference/inference_and_eval.sh`.
## Citation
If you find this project useful in your research, please cite:
```
@article{DBLP:journals/corr/abs-2407-21693,
  author       = {Ming Zhang and
                  Caishuang Huang and
                  Yilong Wu and
                  Shichun Liu and
                  Huiyuan Zheng and
                  Yurui Dong and
                  Yujiong Shen and
                  Shihan Dou and
                  Jun Zhao and
                  Junjie Ye and
                  Qi Zhang and
                  Tao Gui and
                  Xuanjing Huang},
  title        = {TransferTOD: {A} Generalizable Chinese Multi-Domain Task-Oriented
                  Dialogue System with Transfer Capabilities},
  journal      = {CoRR},
  volume       = {abs/2407.21693},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2407.21693},
  doi          = {10.48550/ARXIV.2407.21693},
  eprinttype    = {arXiv},
  eprint       = {2407.21693},
  timestamp    = {Wed, 21 Aug 2024 20:53:27 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2407-21693.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```