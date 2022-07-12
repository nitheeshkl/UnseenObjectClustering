#!/usr/bin/env python3

from ast import parse
from cgi import test
import sys
import os
import argparse
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dataset for UCN from DoPose dataset"
    )
    parser.add_argument("src", help="source dir for DoPose dataset")
    parser.add_argument("dst", help="destination dir to generate the UCN dataset")
    parser.add_argument("split", help="train | test | val")
    parser.add_argument(
        "--visualize",
        "-v",
        help="visualize rgb, depth, and segmask. Does not generate file in dst",
        action="store_true",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    return args


def visualize(rgb, depth, mask):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
    axs[0].imshow(rgb)
    axs[0].set_title("RGB")
    axs[0].axis("off")
    axs[1].imshow(depth)
    axs[1].set_title("Depth")
    axs[1].axis("off")
    axs[2].imshow(mask)
    axs[2].set_title("SegMask")
    axs[2].axis("off")
    plt.show()
    plt.close()


def generate(src, dst, split, visualize=False):

    if not os.path.exists(src):
        print("Error: {} not found".format(src))
        sys.exit(-1)

    print("loading source dataset from ", src)

    src_rgb_dir = os.path.join(src, "bop_data/hope/train_pbr/000000/rgb")
    src_depth_dir = os.path.join(src, "bop_data/hope/train_pbr/000000/depth")
    src_mask_dir = os.path.join(src, "coco_data/images")

    rgb_imgs = sorted(glob.glob(os.path.join(src_rgb_dir, "*.png")))

    print("found {} rgb images".format(len(rgb_imgs)))

    if not os.path.exists(dst):
        print("creating ", dst)
        os.makedirs(dst)
    dst_rgb_dir = dst + "/rgb"
    if not os.path.exists(dst_rgb_dir):
        print("creating ", dst_rgb_dir)
        os.makedirs(dst_rgb_dir)
    dst_depth_dir = dst + "/depth"
    if not os.path.exists(dst_depth_dir):
        print("creating ", dst_depth_dir)
        os.makedirs(dst_depth_dir)
    dst_mask_dir = dst + "/mask"
    if not os.path.exists(dst_mask_dir):
        print("creating ", dst_mask_dir)
        os.makedirs(dst_mask_dir)

    dst_file_indices = []

    pbar = tqdm(rgb_imgs)
    for rgb_file in pbar:
        img_idx = os.path.splitext(os.path.basename(rgb_file))[0]
        rgb_img = np.array(Image.open(rgb_file))
        depth_img = np.array(Image.open(src_depth_dir + "/{}.png".format(img_idx)))
        mask_img = np.array(Image.open(src_mask_dir + "/{}.png".format(img_idx)))

        if visualize:
            visualize(rgb_img, depth_img, mask_img)
        else:
            dst_file_indices.append(img_idx)
            Image.fromarray(rgb_img).save(dst_rgb_dir + "/{}.png".format(img_idx))
            Image.fromarray(depth_img).save(dst_depth_dir + "/{}.png".format(img_idx))
            Image.fromarray(mask_img).save(dst_mask_dir + "/{}.png".format(img_idx))

        pbar.set_description("{}".format(img_idx))
        pbar.refresh()

    if not visualize:
        num_indices = len(dst_file_indices)
        dst_file_indices = np.array(dst_file_indices)
        rand_indices = np.random.permutation(num_indices)
        rand_dst_file_indices = dst_file_indices[rand_indices]

        np.save(dst + "/{}.npy".format(split), rand_dst_file_indices)

        shutil.copyfile(
            os.path.join(src, "bop_data/hope/camera.json"),
            os.path.join(dst, "camera_params.json"),
        )


if __name__ == "__main__":
    args = parse_args()

    generate(args.src, args.dst, args.split, args.visualize)
