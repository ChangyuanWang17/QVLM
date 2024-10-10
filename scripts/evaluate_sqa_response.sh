CUDA_VISIBLE_DEVICES=0 python ../llava/eval/eval_science_qa.py \
    --base-dir /<path>/ScienceQA/data/scienceqa \
    --result-file /<path>/LLaVA/results/ScienceQA/LLaVA-vicuna-7B-v1.3-4bit.jsonl \
    --output-file /<path>/LLaVA/results/ScienceQA/test_llava-7b_output.json \
    --output-result /<path>/LLaVA/results/ScienceQA/test_llava-7b_result.json 