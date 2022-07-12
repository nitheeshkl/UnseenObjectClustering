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
from multiprocessing import Pool, cpu_count

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dataset for UCN from synthetic data generated from blenderproc"
    )
    parser.add_argument("src", help="source dir containing synthetic dataset")
    parser.add_argument("dst", help="destination dir to generate the UCN dataset")
    parser.add_argument("split", help="train | test | val | trainval")
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


def show_imgs(rgb, depth, mask):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3), constrained_layout=True)
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


class Generator:
    
    def __init__(self, src : str, dst : str, split : str, visualize : bool = False):

        if not os.path.exists(src):
            print("Error: {} not found".format(src))
            sys.exit(-1)

        self.src = src
        self.dst = dst
        self.split = split
        self.visualize = visualize


        print("loading source dataset from ", self.src)

        self.src_rgb_dir = os.path.join(self.src, "rgb")
        self.src_depth_dir = os.path.join(self.src, "depth")
        self.src_mask_dir = os.path.join(self.src, "mask")

        print("reading rgb files...")
        self.rgb_imgs = sorted(glob.glob(os.path.join(self.src_rgb_dir, "*.png")))
        print("found {} rgb images".format(len(self.rgb_imgs)))

        if not os.path.exists(self.dst):
            print("creating ", self.dst)
            os.makedirs(self.dst)
        self.dst_rgb_dir = self.dst + "/rgb"
        if not os.path.exists(self.dst_rgb_dir):
            print("creating ", self.dst_rgb_dir)
            os.makedirs(self.dst_rgb_dir)
        self.dst_depth_dir = self.dst + "/depth"
        if not os.path.exists(self.dst_depth_dir):
            print("creating ", self.dst_depth_dir)
            os.makedirs(self.dst_depth_dir)
        self.dst_mask_dir = self.dst + "/mask"
        if not os.path.exists(self.dst_mask_dir):
            print("creating ", self.dst_mask_dir)
            os.makedirs(self.dst_mask_dir)

    def process_file(self, rgb_file: str) -> None:
        img_idx = os.path.splitext(os.path.basename(rgb_file))[0]
        rgb_img = np.array(Image.open(rgb_file))
        mask_img = np.array(Image.open(self.src_mask_dir + "/{}.png".format(img_idx)))
        depth = np.load(self.src_depth_dir + "/{}.npy".format(img_idx))

        background = np.where(mask_img == 1)  # get background mask
        depth[background] = 0  # remove background depth
        mask_img[np.where(mask_img < 4)] = 0  # remove background and container labels

        Image.fromarray(rgb_img).save(self.dst_rgb_dir + "/{}.png".format(img_idx))
        np.save(self.dst_depth_dir + "/{}.npy".format(img_idx), depth)
        Image.fromarray(mask_img).save(self.dst_mask_dir + "/{}.png".format(img_idx))

        return img_idx


    def generate(self):

        if self.visualize:
            # process one by one
            pbar = tqdm(self.rgb_imgs)
            for rgb_file in pbar:
                img_idx = os.path.splitext(os.path.basename(rgb_file))[0]
                rgb_img = np.array(Image.open(rgb_file))
                mask_img = np.array(Image.open(self.src_mask_dir + "/{}.png".format(img_idx)))
                depth = np.load(self.src_depth_dir + "/{}.npy".format(img_idx))

                background = np.where(mask_img == 1)  # get background mask
                depth[background] = 0  # remove background depth
                mask_img[np.where(mask_img < 4)] = 0  # remove background and container labels

                show_imgs(rgb_img, depth, mask_img)

                pbar.set_description("{}".format(img_idx))
                pbar.refresh()
        else:

            dst_file_indices = []
            with Pool(processes=cpu_count()) as pool:
                for idx in tqdm(pool.imap(self.process_file, self.rgb_imgs), total=len(self.rgb_imgs)):
                    dst_file_indices.append(idx)

            if len(dst_file_indices) != len(self.rgb_imgs):
                print("missed {} files".format(len(dst_file_indices) - len(self.rgb_imgs)))

            # create splits
            print("creating {} split...".format(self.split))
            num_indices = len(dst_file_indices)
            dst_file_indices = np.array(dst_file_indices)
            rand_indices = np.random.permutation(num_indices)
            rand_dst_file_indices = dst_file_indices[rand_indices]

            if self.split == "trainval":
                # if trainval split, then create separate train and val splits
                num_train_indices = int(num_indices * 0.9)
                np.save(self.dst + "/train.npy", rand_dst_file_indices[:num_train_indices])
                np.save(self.dst + "/val.npy", rand_dst_file_indices[num_train_indices:])
            else:
                np.save(self.dst + "/{}.npy".format(self.split), rand_dst_file_indices)

            shutil.copyfile(
                os.path.join(self.src, "camera.json"),
                os.path.join(self.dst, "camera_params.json"),
            )


if __name__ == "__main__":
    args = parse_args()

    generator = Generator(args.src, args.dst, args.split, args.visualize)
    generator.generate()
