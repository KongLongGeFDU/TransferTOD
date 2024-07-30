MODEL_NAME_OR_PATH=base_model #baichuan-inc/Baichuan-7B-Base
MODEL_NAME=your_model

DATA_PATH_TRAIN=./data/fine_tune_1/4_clean_noise_gpt_human/train.json
OUTPUT_PATH=./models/$MODEL_NAME
mkdir -p $OUTPUT_PATH

OUTPUT_LOG_PATH=./fine_tune/log
mkdir -p $OUTPUT_LOG_PATH

hostfile=""
master_port=29501
localhost=0,1
model_max_length=1024
deepspeed --hostfile=$hostfile --master_port $master_port --include localhost:$localhost ./fine_tune/fine-tune.py \
    --report_to "none" \
    --data_path $DATA_PATH_TRAIN \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_PATH \
    --model_max_length $model_max_length \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --learning_rate 9.65e-6 \
    --lr_scheduler_type cosine \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed ./fine_tune/ds_config.json \
    --bf16 True \
    --tf32 True \
    --use_lora True \
    2>&1 | tee "$OUTPUT_LOG_PATH/$MODEL_NAME.log" | tee "$OUTPUT_PATH/training.log"
    # &> $OUTPUT_LOG_PATH/training.log



    