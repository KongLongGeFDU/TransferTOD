# TransferTOD
The code repository of paper "TransferTOD: A Generalizable Chinese Multi-Domain Task-Oriented Dialogue System with Transfer Capabilities"

arxiv: https://arxiv.org/abs/2407.21693

> All the data used in two-staged finetuning and the raw data of TransferTOD is included in directory `./data`. For each version, `train.json` is a mixed data of `train_slot.json` and equivalent amounts of `./data/raw_data/belle_data/belle_filtered_950k_train.jsonl`
>
> For full fine-tuning, run `./fine_tune/scripts/finetune_full.sh`, while for lora fine-tuning, run `./fine_tune/scripts/finetune_lora.sh`.
>
> For inference and evaluation with the TransferTOD test set, run `./inference/inference_and_eval.sh`.

Contact Us:
mingzhang23@m.fudan.edu.cn
