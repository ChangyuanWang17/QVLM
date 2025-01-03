# [NeurIPS'24]Q-VLM: Post-training Quantization for Large Vision-Language Models

*Efficient and accurate memory saving method towards W4A4 large multi-modal models.* [[Paper](https://arxiv.org/abs/2410.08119)]

> Q-VLM: Post-training Quantization for Large Vision-Language Models  
> [Changyuan Wang](https://changyuanwang17.github.io), [Ziwei Wang](https://ziweiwangthu.github.io), [Xiuwei Xu](https://xuxw98.github.io/), [Yansong Tang](https://andytang15.github.io), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/)


## Install

1. Clone this repository and navigate to QVLM folder
```bash
git clone https://github.com/ChangyuanWang17/QVLM.git
cd QVLM
```

2. Install Package
```Shell
conda create -n QVLM python=3.10 -y
conda activate QVLM
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for Q-VLM
```Shell
pip uninstall bitsandbytes
cd custom_bitsandbytes
python setup.py install
```

## Generate and evaluate SQA response
The following experiments were performed in GeForce RTX 3090 with 24GB memory.
```Shell
sh scripts/generate_sqa_response.sh
sh scripts/evaluate_sqa_response.sh
```
Generate and evaluate with multi GPUs
```Shell
sh scripts/generate_sqa_response_multi.sh
sh scripts/evaluate_sqa_response_multi.sh
```

## Pretrained LVLM Weights
Please check out [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) for all public LLaVA checkpoints, and the instructions of how to use the weights. Thanks for LLaVA (https://github.com/haotian-liu/LLaVA) for the amazing open-source model!

We also uploaded [LLaVA-v1.3-7B](https://huggingface.co/ChangyuanWang/LLaVA-vicuna-7B-v1.3-ScienceQA) model finetuned on ScienceQA dataset to test the effect of quantization.

## ScienceQA

Please check out the documentation [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/ScienceQA.md).


## Acknowledgement
We thank the authors of following works for opening source their excellent codes.
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)

