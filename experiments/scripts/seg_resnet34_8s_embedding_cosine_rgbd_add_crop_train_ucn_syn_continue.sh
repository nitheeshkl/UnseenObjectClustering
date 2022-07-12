#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

./tools/trainval_net.py \
  --network seg_resnet34_8s_embedding \
  --pretrained output/ucn_syn_dataset/ucn_syn_dataset_train/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth \
  --startepoch 16 \
  --dataset ucn_syn_dataset_train \
  --dataset-val ucn_syn_dataset_val \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_ucn_syn.yml \
  --solver adam \
  --epochs 32
