#!/usr/bin/env python3

from cgi import test
import sys
import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate dataset for UCN from DoPose dataset")
    parser.add_argument("src", help="source dir for DoPose dataset")
    parser.add_argument("dst", help="destination dir to generate the UCN dataset")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    return args

def visualize(rgb, depth, mask):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,3))
    axs[0].imshow(rgb)
    axs[0].set_title("RGB")
    axs[0].axis('off')
    axs[1].imshow(depth)
    axs[1].set_title("Depth")
    axs[1].axis('off')
    axs[2].imshow(mask)
    axs[2].set_title("SegMask")
    axs[2].axis('off')
    plt.show()
    plt.close()

def generate(src, dst):

    if not os.path.exists(src):
        print("Error: {} not found".format(src))
        sys.exit(-1)

    print("loading source dataset from ", src)

    scene_dirs = sorted(glob.glob(src + "/*"))

    print("found {} scenes".format(len(scene_dirs)))

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

    pbar = tqdm(scene_dirs)
    for scene_dir in pbar:
        scene_idx = os.path.basename(scene_dir)
        rgb_views = sorted(glob.glob(scene_dir + "/rgb/*.png"))

        for view in rgb_views:
            view_idx = os.path.splitext(os.path.basename(view))[0]
            rgb_img = np.array(Image.open(scene_dir + "/rgb/{}.png".format(view_idx)))
            depth_img = np.array(Image.open(scene_dir + "/depth/{}.png".format(view_idx)))
            masks = sorted(glob.glob(scene_dir + "/mask_visib/{}_*.png".format(view_idx)))
            seg_mask = np.zeros_like(depth_img)

            for i, mask in enumerate(masks):
                m = (np.array(Image.open(mask)) // 255) * (i + 1)
                seg_mask += m

            #visualize(rgb_img, depth_img, seg_mask)

            dst_file_idx = "{}_{}".format(scene_idx, view_idx)
            dst_file_indices.append(dst_file_idx)
            pbar.set_description("{}".format(dst_file_idx))
            pbar.refresh()

            Image.fromarray(rgb_img).save(dst_rgb_dir + "/{}.png".format(dst_file_idx))
            Image.fromarray(depth_img).save(dst_depth_dir + "/{}.png".format(dst_file_idx))
            Image.fromarray(seg_mask).save(dst_mask_dir + "/{}.png".format(dst_file_idx))

    num_indices = len(dst_file_indices)
    dst_file_indices = np.array(dst_file_indices)
    rand_indices = np.random.permutation(num_indices)
    rand_dst_file_indices = dst_file_indices[rand_indices]
    train_split = int(num_indices * 0.6)
    test_split = int(num_indices * 0.2)
    train_indices = rand_dst_file_indices[:train_split]
    test_indices = rand_dst_file_indices[-test_split:]
    val_indices = rand_dst_file_indices[train_split : -test_split ]

    np.save(dst + "/train.npy", train_indices)
    np.save(dst + "/val.npy", val_indices)
    np.save(dst + "/test.npy", test_indices)

    # TODO: create camera_params.json containing camera intrinsics 

if __name__ == "__main__":
    args = parse_args()

    generate(args.src, args.dst)
