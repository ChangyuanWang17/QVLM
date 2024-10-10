#!/bin/bash

CHUNKS=8
output_file="/<path>/LLaVA/results/ScienceQA/llava-7b-v1.3-6epoch-awq-4bit.jsonl"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for idx in $(seq 0 $((CHUNKS-1))); do
  cat "/home/ubuntu/wcy/LLaVA/LLaVA/results/ScienceQA/LLaVA-vicuna-7B-v1.3-4bit-chunk${idx}.jsonl" >> "$output_file"
done

CUDA_VISIBLE_DEVICES=7 python ../llava/eval/eval_science_qa.py \
    --base-dir /<path>/ScienceQA/data/scienceqa \
    --result-file $output_file \
    --output-file /<path>/LLaVA/results/ScienceQA/test_llava-7b_output.json \
    --output-result /<path>/LLaVA/results/ScienceQA/test_llava-7b_result.json 

    