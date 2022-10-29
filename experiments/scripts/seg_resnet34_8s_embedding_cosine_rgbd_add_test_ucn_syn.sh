#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

./tools/test_net.py \
  --network seg_resnet34_8s_embedding \
  --pretrained mujin_model_checkpoints/UCN/FineTuned_at_Mujin/ucn_syn_dataset_train_latest_end_of_july/ucn_syn_seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_32.checkpoint.pth \
  --dataset ucn_syn_dataset_test \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_ucn_syn.yml
