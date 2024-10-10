# [NeurIPS'24]Q-VLM: Post-training Quantization for Large Vision-Language Models

*Efficient and accurate memory saving method towards W4A4 large multi-modal models.* [[Paper](https://arxiv.org/abs/)]


## Install

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd QVLM
```

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
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

## Pretrained LLaVA Weights
Please check out our [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) for all public LLaVA checkpoints, and the instructions of how to use the weights.

## ScienceQA

Please check out the documentation [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/ScienceQA.md).


## Acknowledgement
We thank the authors of following works for opening source their excellent codes.
- [LLaVA](https://github.com/haotian-liu/LLaVA)

