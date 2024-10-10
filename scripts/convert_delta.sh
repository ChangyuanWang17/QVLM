CUDA_VISIBLE_DEVICES=1 python3 -m llava.model.apply_delta \
    --base /home/ubuntu/lyn/LLaVA/pyllama/converted_meta/7B \
    --target /home/ubuntu/wcy/LLaVA/pretrain/LLaVA-7B-v0-own \
    --delta /home/ubuntu/wcy/LLaVA/pretrain/LLaVA-7B-delta-v0