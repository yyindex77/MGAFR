# Multiplex Graph Aggregation and Feature Refinement for Unsupervised Incomplete Multimodal Emotion Recognition

## Dataset Preparation

Please refer to [this link](https://github.com/zeroQiaoba/GCNet) for instructions on dataset preparation.

## Run MGAFR

```bash
CUDA_VISIBLE_DEVICES=0 python -W ignore -u train_mgafr.py --mask-type='constant-0.0' --dataset='CMUMOSEI'
