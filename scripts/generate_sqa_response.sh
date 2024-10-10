CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_science \
    --model-path <model-path> \
    --question-file /<path>/ScienceQA/data/scienceqa/llava_test_QCM-LEPA.json \
    --image-folder /<path>/ScienceQA/data/scienceqa/images/test \
    --question-file-calibrate /<path>/ScienceQA/data/scienceqa/llava_train_QCM-LEPA.json \
    --image-folder-calibrate /<path>/ScienceQA/data/scienceqa/images/train \
    --answers-file /<path>/LLaVA/results/ScienceQA/LLaVA-vicuna-7B-v1.3-4bit.jsonl \
    --conv-mode llava_v1 \
    --load-4bit