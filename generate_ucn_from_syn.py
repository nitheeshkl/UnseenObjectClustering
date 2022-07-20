#!/usr/bin/env python3

from ast import parse
from cgi import test
import sys
import os
import argparse
import glob
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
from mpl_toolkits.axes_grid1 import make_axes_locatable
import open3d as o3d
import cv2

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

def show_depth(depth):
    print("depth min={}, max={}".format(depth.min(), depth.max()))
    plt.close('all')
    fig = plt.figure()
    im = plt.imshow(depth, interpolation="none")

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes('right', '5%', pad='3%')
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
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
        self.downscale = True # resize data to 640x480


        print("loading source dataset from ", self.src)

        self.src_rgb_dir = os.path.join(self.src, "rgb")
        self.src_depth_dir = os.path.join(self.src, "depth")
        self.src_mask_dir = os.path.join(self.src, "mask")
        self.src_cam_pose_dir = os.path.join(self.src, "cam_pose")

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
        self.dst_depth_img_dir = os.path.join(self.dst, "depth_img")
        if not os.path.exists(self.dst_depth_img_dir):
            print("creating ", self.dst_depth_img_dir)
            os.makedirs(self.dst_depth_img_dir)

    def compute_xyz(self, depth, camera_params, scaled=False):
        height = depth.shape[0]
        width = depth.shape[1]
        img_height = camera_params['height']
        img_width = camera_params['width']
        fx = camera_params['fx']
        fy = camera_params['fy']
        if "x_offset" in camera_params.keys():
            px = camera_params['x_offset']
            py = camera_params['y_offset']
        else:
            px = camera_params['cx']
            py = camera_params['cy']

        indices =  np.indices((height, width), dtype=np.float32).transpose(1,2,0) #[H,W,2]

        if scaled:
            scale_x = width / img_width
            scale_y = height / img_height
        else:
            scale_x, scale_y = 1., 1.

        # print("scale = ({},{})".format(scale_x, scale_y))

        fx, fy = fx * scale_x, fy * scale_y
        px, py = px * scale_x, py * scale_y

        z = depth
        x = (indices[..., 1] - px) * z / fx
        y = (indices[..., 0] - py) * z / fy
        xyz_img = np.stack([x,y,z], axis=-1) # [H,W,3]

        return xyz_img

    def __get_color(self, img_idx: str) -> np.ndarray:
        rgb_img = np.array(Image.open(self.src_rgb_dir +"/{}.png".format(img_idx)))

        if self.downscale:
            rgb_img = cv2.resize(rgb_img[120:,:,:], (640, 480))

        return rgb_img

    def __get_mask(self, img_idx: str) -> np.ndarray:
        mask_img = np.array(Image.open(self.src_mask_dir + "/{}.png".format(img_idx)))

        if self.downscale:
            mask_img = cv2.resize(mask_img[120:,:], (640, 480), interpolation=cv2.INTER_NEAREST)

        return mask_img

    def __get_points(self, img_idx:str ) -> np.ndarray:

        depth = np.load(self.src_depth_dir + "/{}.npy".format(img_idx))
        depth = depth / 1000. # mm to m
        cam_pose = np.load(self.src_cam_pose_dir + "/{}.npy".format(img_idx))
        cam_params = None
        with open(self.src + "/camera.json") as f:
            cam_params = json.load(f)

        cam_loc = cam_pose[:3, 3]

        H, W = depth.shape

        # create o3d point cloud structure
        structured_points = self.compute_xyz(depth, cam_params)
        points = structured_points.reshape(-1,3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # compute normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=15))
        norm_pcd = pcd.normalize_normals()
        normals = np.asarray(norm_pcd.normals).reshape(H, W, 3)

        # remove planes that are close to parallel to camera view
        t = -cam_loc
        unit_vec = t / np.linalg.norm(t)
        dot_prod = np.abs(np.dot(normals, unit_vec))
        mask = (dot_prod > np.cos(np.radians(75))).astype(float)

        result = (structured_points * mask.reshape(H,W,1)).astype(structured_points.dtype)

        if self.downscale:
            result = cv2.resize(result, (640, 480), interpolation=cv2.INTER_NEAREST)

        return result

    def __get_data(self, img_idx: str) -> tuple:
        rgb_img = self.__get_color(img_idx)
        mask_img = self.__get_mask(img_idx)
        points = self.__get_points(img_idx)

        H, W = mask_img.shape
        fg_mask = (mask_img!=1).astype(float).reshape(H,W,1) # create foreground mask including container

        # filter background from rgb and depth
        rgb_img = (rgb_img * fg_mask).astype(rgb_img.dtype)
        points = (points * fg_mask).astype(points.dtype)
        points[np.where(points[:,:,2] > 2.)] = [0, 0, 0]
        # remove background and container from label mask
        mask_img[np.where(mask_img < 4)] = 0

        return rgb_img, mask_img, points

    def process_file(self, rgb_file: str) -> None:
        img_idx = os.path.splitext(os.path.basename(rgb_file))[0]

        rgb_img, mask_img, points = self.__get_data(img_idx)

        Image.fromarray(rgb_img).save(self.dst_rgb_dir + "/{}.png".format(img_idx))
        np.save(self.dst_depth_dir + "/{}.npy".format(img_idx), points)
        Image.fromarray(mask_img).save(self.dst_mask_dir + "/{}.png".format(img_idx))

        plt.figure()
        plt.tight_layout()
        plt.imshow(points[:,:,2])
        plt.savefig(self.dst_depth_img_dir + "/{}.png".format(img_idx))
        plt.close()

        return img_idx


    def generate(self):

        if self.visualize:
            # process one by one
            pbar = tqdm(self.rgb_imgs)
            for rgb_file in pbar:
                img_idx = os.path.splitext(os.path.basename(rgb_file))[0]

                rgb_img, mask_img, points = self.__get_data(img_idx)

                show_imgs(rgb_img, points[:,:,2], mask_img)
                # show_depth(partial_depth)

                pbar.set_description("{}".format(img_idx))
                pbar.refresh()
        else:

            dst_file_indices = []
            # num_cpus = cpu_count()
            num_cpus = 8
            with Pool(processes=num_cpus) as pool:
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
