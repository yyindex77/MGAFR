# Multiplex Graph Aggregation and Feature Refinement for Unsupervised Incomplete Multimodal Emotion Recognition

## Dataset Preparation

Please refer to [this link](https://github.com/zeroQiaoba/GCNet) for instructions on dataset preparation.

## How to Run

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_gcnet.py --mask-type='constant-0.0' --dataset='CMUMOSEI'
