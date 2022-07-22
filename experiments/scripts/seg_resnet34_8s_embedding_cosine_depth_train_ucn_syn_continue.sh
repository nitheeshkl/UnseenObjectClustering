#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

./tools/train_net.py \
  --network seg_resnet34_8s_embedding \
  --pretrained data/checkpoints/seg_resnet34_8s_embedding_cosine_depth_sampling_epoch_16.checkpoint.pth \
  --startepoch 16 \
  --dataset ucn_syn_dataset_train \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_depth_ucn_syn.yml \
  --solver adam \
  --epochs 32
