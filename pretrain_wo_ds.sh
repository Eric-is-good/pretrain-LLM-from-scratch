#!/bin/bash

python train/training.py \
    --model_name_or_path HolmesLLMXS/ \
    --data_path /root/autodl-fs/Holmes_data/1 \
    --data_cache_path /root/autodl-tmp \
    --bf16 True \
    --output_dir XS_out/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1 \
    --save_total_limit 2 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine_with_restarts" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --report_to wandb
