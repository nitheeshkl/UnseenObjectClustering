#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

./tools/train_net.py \
  --network seg_resnet34_8s_embedding \
  --dataset dopose_dataset_train \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_dopose.yml \
  --solver adam \
  --epochs 16
