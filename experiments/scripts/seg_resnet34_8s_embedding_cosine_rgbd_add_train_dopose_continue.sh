#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

./tools/train_net.py \
  --network seg_resnet34_8s_embedding \
  --pretrained output/dopose_dataset/dopose_dataset_train/seg_resnet34_8s_embedding_cosine_rgbd_sampling_epoch_16.checkpoint.pth \
  --startepoch 16 \
  --dataset dopose_dataset_train \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_dopose.yml \
  --solver adam \
  --epochs 16
