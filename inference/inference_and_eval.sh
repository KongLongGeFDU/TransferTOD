export CUDA_VISIBLE_DEVICES=0
base=./Baichuan2-7B-Base
data_types=("clean" "noise")

model=your_model
ckpt=./models/$model

for data_type in ${data_types[@]}; do
    mkdir -p ./results/${model}
    mkdir -p ./logs/${model}
    mkdir -p ./eval/${model}

    python inference.py \
        --lora \
        --base $base\
        --checkpoint $ckpt\
        --test_data ./data/fine_tune_2/30_courier/test/test_${data_type}_ood_30.json\
        --result_file results/${model}/${model}_${data_type}_ood_30.jsonl\
        > logs/${model}/${model}_${data_type}_ood_30.log 2>&1
    python eval.py \
        --actual_file ./data/fine_tune_2/30_courier/test/test_${data_type}_ood_30.json\
        --predict_file results/${model}/${model}_${data_type}_ood_30.jsonl\
        > eval/${model}/${model}_${data_type}_ood_30.log
done