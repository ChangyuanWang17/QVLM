CUDA_VISIBLE_DEVICES=1 python -m llava.serve.cli \
    --model-path /home/ubuntu/wcy/LLaVA/pretrain/LLaVA-13b-delta-v0-science_qa \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --load-8bit