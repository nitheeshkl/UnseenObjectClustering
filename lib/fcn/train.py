# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.nn as nn
import time
import sys, os
import numpy as np
import matplotlib.pyplot as plt

from fcn.config import cfg
from fcn.test_common import _vis_minibatch_segmentation

import wandb


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return "{:.3f} ({:.3f})".format(self.val, self.avg)

def normalize(mat):
    min, max = mat.min(), mat.max()

    norm = np.clip(mat, min, max)
    e = 1e-10
    scale = (max - min) + e
    norm = (norm - min) / scale

    return norm

def normalize_descriptor(res, stats=None):
    """
    Normalizes the descriptor into RGB color space
    :param res: numpy.array [H,W,D]
        Output of the network, per-pixel dense descriptor
    :param stats: dict, with fields ['min', 'max', 'mean'], which are used to normalize descriptor
    :return: numpy.array
        normalized descriptor
    """

    if stats is None:
        res_min = res.min()
        res_max = res.max()
    else:
        res_min = np.array(stats['min'])
        res_max = np.array(stats['max'])

    normed_res = np.clip(res, res_min, res_max)
    eps = 1e-10
    scale = (res_max - res_min) + eps
    normed_res = (normed_res - res_min) / scale
    return normed_res

def feature_tensor_to_img(features):
    print(features.shape)
    i = 0
    height, width = features.shape[-2:]
    channels = 3
    im_feature = torch.cuda.FloatTensor(height, width, channels)
    for j in range(channels):
        im_feature[:, :, j] = torch.sum(features[i, j::channels, :, :], dim=0)
    im_feature = normalize_descriptor(im_feature.detach().cpu().numpy())
    im_feature *= 255
    im_feature = im_feature.astype(np.uint8)
    return im_feature

def train_segnet(train_loader, network, optimizer, epoch):

    batch_time = AverageMeter()
    epoch_size = len(train_loader)

    # switch to train mode
    network.train()

    for i, sample in enumerate(train_loader):

        end = time.time()

        # construct input
        image = sample["image_color"].cuda()
        if cfg.INPUT == "DEPTH" or cfg.INPUT == "RGBD":
            depth = sample["depth"].cuda()
        else:
            depth = None

        label = sample["label"].cuda()
        loss, intra_cluster_loss, inter_cluster_loss, features = network(
            image, label, depth
        )
        loss = torch.sum(loss)
        intra_cluster_loss = torch.sum(intra_cluster_loss)
        inter_cluster_loss = torch.sum(inter_cluster_loss)
        out_label = None

        if cfg.TRAIN.VISUALIZE:
            _vis_minibatch_segmentation(
                image, depth, label, out_label, features=features
            )

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()

        rgb_img = (
            (sample["image_color"][0].permute([1, 2, 0]) + pixel_mean)
            .detach()
            .cpu()
            .numpy()
        )
        points = sample["depth"][0].permute([1, 2, 0]).detach().cpu().numpy()
        points = points.reshape(-1, 3)
        mask = sample["label"][0].permute([1, 2, 0]).detach().cpu().numpy()
        rgb_mask = mask.squeeze(2) + 1
        points_mask = mask.reshape(-1, 1) + 2
        points = np.concatenate([points, points_mask], axis=1)

        wandb_img = wandb.Image(rgb_img, masks= {
            "GT": {
                "mask_data": rgb_mask
            }
        })

        features_img = feature_tensor_to_img(features)

        wandb.log(
            {
                "train_loss": loss,
                "intra_cluster_loss": intra_cluster_loss,
                "inter_cluster_loss": inter_cluster_loss,
                "epoch": epoch,
                "image": wandb_img,
                "features": wandb.Image(features_img),
                "point_cloud": wandb.Object3D(points),
            }
        )

        print(
            "[%d/%d][%d/%d], loss %.4f, loss intra: %.4f, loss_inter %.4f, lr %.6f, time %.2f"
            % (
                epoch,
                cfg.epochs,
                i,
                epoch_size,
                loss,
                intra_cluster_loss,
                inter_cluster_loss,
                optimizer.param_groups[0]["lr"],
                batch_time.val,
            )
        )
        cfg.TRAIN.ITERS += 1
