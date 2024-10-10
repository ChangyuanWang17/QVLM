#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
# PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################
PROMPT_VERSION=v0
#!/bin/bash

deepspeed --include localhost:0,1  \
    ../llava/train/train_mem.py \
    --deepspeed ./zero3_offload.json \
    --model_name_or_path /home/ubuntu/lyn/LLaVA/LLaVA/pretrain/vicuna \
    --version $PROMPT_VERSION \
    --data_path /home/ubuntu/wcy/LLaVA/ScienceQA/data/scienceqa/llava_train_QCM-LEPA.json \
    --image_folder /home/ubuntu/wcy/LLaVA/ScienceQA/data/scienceqa/images/train \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter /home/ubuntu/wcy/LLaVA/pretrain/projector/LLaVA-vicuna-7B-v1.3/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /home/ubuntu/wcy/LLaVA/LLaVA/results/checkpoints/llava-vicuna-7b-v0-pretrain_cc595k_plain-ScienceQA_QCM_LEA-12e \
    --num_train_epochs 12 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb 