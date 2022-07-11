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

        wandb.log(
            {
                "train_loss": loss,
                "intra_cluster_loss": intra_cluster_loss,
                "inter_cluster_loss": inter_cluster_loss,
                "epoch": epoch,
                "image": wandb_img,
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
