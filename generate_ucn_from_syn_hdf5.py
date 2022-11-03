#!/usr/bin/env python3

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
from multiprocessing import Pool, cpu_count, set_start_method, get_context
from mpl_toolkits.axes_grid1 import make_axes_locatable
import open3d as o3d
import cv2
import h5py


objID_to_modelName = {
    "hope_3": "butter",
    "hope_6": "cookies",
    "hope_8": "cheese",
    "hope_12": "mac_cheese",
    "hope_23": "raisins",
    "hope_5": "pudding",
    "hope_9": "granola",
    "hope_22": "popcorn",
    "hope_25": "spaghetti",
    "ruapc_2": "cheez_it",
    "ycbv_2": "cheez_it_deformed",
    "ruapc_3": "crayola",
    "ycbv_3": "domino_deformed",
    "ruapc_5": "expo_eraser",
    "hb_18": "jaffa_cakes_deformed",
    "ycbv_7": "jello_chocolate_deformed",
    "ycbv_8": "jello_strawberry_deformed",
    "hb_29": "nelson_tea",
    "ruapc_12": "oreo",
    "ruapc_13": "papermate_pen",
    "tyol_14": "plain_cracker",
    "ruapc_1": "spark_plug",
    "ruapc_8": "stick_straw",
    "ruapc_9": "sticky_notes",
    "tyol_2": "sudoku_book",
}

modelName_to_objID = {
    "butter": "hope_3",
    "cookies": "hope_6",
    "cheese": "hope_8",
    "mac_cheese": "hope_12",
    "raisins": "hope_23",
    "pudding": "hope_5",
    "granola": "hope_9",
    "popcorn": "hope_22",
    "spaghetti": "hope_25",
    "cheez_it": "ruapc_2",
    "cheez_it_deformed": "ycbv_2",
    "crayola": "ruapc_3",
    "domino_deformed": "ycbv_3",
    "expo_eraser": "ruapc_5",
    "jaffa_cakes_deformed": "hb_18",
    "jello_chocolate_deformed": "ycbv_7",
    "jello_strawberry_deformed": "ycbv_8",
    "nelson_tea": "hb_29",
    "oreo": "ruapc_12",
    "papermate_pen": "ruapc_13",
    "plain_cracker": "tyol_14",
    "spark_plug": "ruapc_1",
    "stick_straw": "ruapc_8",
    "sticky_notes": "ruapc_9",
    "sudoku_book": "tyol_2",
}

train_objects = [
    "butter",
    "cookies",
    "cheese",
    "mac_cheese",
    "raisins",
    "pudding",
    "granola",
    "popcorn",
    "spaghetti",
]
val_objects = [
    "cheez_it_deformed",
    "jello_chocolate_deformed",
    "expo_eraser",
    "nelson_tea",
]
test_objects = [
    "cheez_it",
    "crayola",
    "domino_deformed",
    "jaffa_cakes_deformed",
    "jello_strawberry_deformed",
    "oreo",
    "papermate_pen",
    "plain_cracker",
    "spark_plug",
    "stick_straw",
    "sticky_notes",
    "sudoku_book",
]


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


def visualize_imgs(rgb, depth, mask, show=False, saveFile=None):

    plt.close("all")
    fig = plt.figure(figsize=(24, 6))
    # rgb
    ax1 = fig.add_subplot(131)
    ax1.imshow(rgb)
    ax1.set_title("Color")
    ax1.axis("off")
    # depth
    ax2 = fig.add_subplot(132)
    im = ax2.imshow(depth, interpolation="None", cmap="turbo")
    ax2.set_title("Depth")
    ax2.axis("off")
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0)
    fig.colorbar(im, cax=cax, orientation="vertical")
    # mask
    ax2 = fig.add_subplot(133)
    ax2.imshow(mask, interpolation="none", cmap="jet")
    ax2.set_title("Instance Mask")
    ax2.axis("off")

    plt.tight_layout(pad=1.5)
    if saveFile is not None:
        plt.savefig(saveFile, dpi=300)
    if show:
        plt.show()
    plt.close()


def visualize_depth(depth, show=False, saveFile=None, showColorbar=False):
    # print("depth min={}, max={}".format(depth.min(), depth.max()))

    plt.close("all")
    fig = plt.figure()
    im = plt.imshow(depth, interpolation="none", cmap="turbo")
    plt.axis("off")
    if showColorbar:
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        plt.colorbar(im, cax=cax)
    plt.tight_layout(pad=0)
    if saveFile is not None:
        plt.savefig(saveFile)
    if show:
        plt.show()
    plt.close()


class Generator:
    def __init__(self, src: str, dst: str, split: str, visualize: bool = False):

        if not os.path.exists(src):
            print("Error: {} not found".format(src))
            sys.exit(-1)

        self.src = src
        self.dst = dst
        self.split = split
        self.visualize = visualize
        self.downscale = False  # resize data to 640x480

        print("loading source dataset from ", self.src)

        self.src_hdf5_dir = os.path.join(self.src, "hdf5")
        self.src_rgb_dir = os.path.join(self.src, "rgb")
        self.src_depth_dir = os.path.join(self.src, "depth")
        self.src_mask_dir = os.path.join(self.src, "mask")
        self.src_cam_pose_dir = os.path.join(self.src, "cam_pose")

        print("reading files...")
        objects = []
        if split == "train":
            objects = train_objects
        elif split == "val":
            objects = val_objects
        elif split == "test":
            objects = test_objects

        random_files = []
        ordered_files = []
        for obj in objects:
            random_files += sorted(
                glob.glob(
                    os.path.join(
                        self.src_hdf5_dir,
                        modelName_to_objID[obj],
                        "random/*/*.hdf5",
                    )
                )
            )
            ordered_files += sorted(
                glob.glob(
                    os.path.join(
                        self.src_hdf5_dir,
                        modelName_to_objID[obj],
                        "ordered/*/*.hdf5",
                    )
                )
            )
        self.all_files = random_files + ordered_files

        # self.rgb_imgs = sorted(glob.glob(os.path.join(self.src_rgb_dir, "*.png")))
        print("found {} hdf5 files".format(len(self.all_files)))

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
        self.dst_visualization_dir = os.path.join(self.dst, "visualization")
        if not os.path.exists(self.dst_visualization_dir):
            print("creating ", self.dst_visualization_dir)
            os.makedirs(self.dst_visualization_dir)

    def compute_xyz(self, depth, camera_params, scaled=False):
        height = depth.shape[0]
        width = depth.shape[1]
        img_height = camera_params["height"]
        img_width = camera_params["width"]
        fx = camera_params["fx"]
        fy = camera_params["fy"]
        if "x_offset" in camera_params.keys():
            px = camera_params["x_offset"]
            py = camera_params["y_offset"]
        else:
            px = camera_params["cx"]
            py = camera_params["cy"]

        indices = np.indices((height, width), dtype=np.float32).transpose(
            1, 2, 0
        )  # [H,W,2]

        if scaled:
            scale_x = width / img_width
            scale_y = height / img_height
        else:
            scale_x, scale_y = 1.0, 1.0

        # print("scale = ({},{})".format(scale_x, scale_y))

        fx, fy = fx * scale_x, fy * scale_y
        px, py = px * scale_x, py * scale_y

        z = depth
        x = (indices[..., 1] - px) * z / fx
        y = (indices[..., 0] - py) * z / fy
        xyz_img = np.stack([x, y, z], axis=-1)  # [H,W,3]

        return xyz_img

    def __get_color(self, img_idx: str) -> np.ndarray:
        rgb_img = np.array(Image.open(self.src_rgb_dir + "/{}.png".format(img_idx)))

        if self.downscale:
            rgb_img = cv2.resize(rgb_img[120:, :, :], (640, 480))

        return rgb_img

    def __get_mask(self, img_idx: str) -> np.ndarray:
        mask_img = np.array(Image.open(self.src_mask_dir + "/{}.png".format(img_idx)))

        if self.downscale:
            mask_img = cv2.resize(
                mask_img[120:, :], (640, 480), interpolation=cv2.INTER_NEAREST
            )

        return mask_img

    def __depth_to_points(self, depth, cam_pose) -> np.ndarray:
        cam_params = None
        with open(self.src + "/camera.json") as f:
            cam_params = json.load(f)

        cam_loc = cam_pose[:3, 3]

        H, W = depth.shape

        # create o3d point cloud structure
        structured_points = self.compute_xyz(depth, cam_params)
        points = structured_points.reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # compute normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=15)
        )
        norm_pcd = pcd.normalize_normals()
        normals = np.asarray(norm_pcd.normals).reshape(H, W, 3)

        # remove planes that are close to parallel to camera view
        t = -cam_loc
        unit_vec = t / np.linalg.norm(t)
        dot_prod = np.abs(np.dot(normals, unit_vec))
        mask = (dot_prod > np.cos(np.radians(75))).astype(float)

        result = (structured_points * mask.reshape(H, W, 1)).astype(
            structured_points.dtype
        )

        if self.downscale:
            result = cv2.resize(result, (640, 480), interpolation=cv2.INTER_NEAREST)

        return result

    def __get_points(self, img_idx: str) -> np.ndarray:

        depth = np.load(self.src_depth_dir + "/{}.npy".format(img_idx))
        depth = depth / 1000.0  # mm to m
        cam_pose = np.load(self.src_cam_pose_dir + "/{}.npy".format(img_idx))

        points = self.__depth_to_points(depth, cam_pose)

        return points

    def __get_data(self, img_idx: str) -> tuple:
        rgb_img = self.__get_color(img_idx)
        mask_img = self.__get_mask(img_idx)
        points = self.__get_points(img_idx)

        H, W = mask_img.shape
        fg_mask = (
            (mask_img != 1).astype(float).reshape(H, W, 1)
        )  # create foreground mask including container

        # filter background from rgb and depth
        rgb_img = (rgb_img * fg_mask).astype(rgb_img.dtype)
        points = (points * fg_mask).astype(points.dtype)
        points[np.where(points[:, :, 2] > 2.0)] = [0, 0, 0]
        # remove background and container from label mask
        mask_img[np.where(mask_img < 4)] = 0

        return rgb_img, mask_img, points

    def __get_data_from_hdf5(self, hdf5_file: str) -> tuple:
        with h5py.File(hdf5_file) as f:
            rgb_img = np.asarray(f["colors"])
            depth = np.asarray(f["depth"])
            mask_img = np.asarray(f["instance_segmaps"])
            cam_pose = np.asarray(f["cam_pose"])
            points = self.__depth_to_points(depth, cam_pose)

        H, W = mask_img.shape
        fg_mask = (
            (mask_img != 1).astype(float).reshape(H, W, 1)
        )  # create foreground mask including container

        # filter background from rgb and depth
        rgb_img = (rgb_img * fg_mask).astype(rgb_img.dtype)
        points = (points * fg_mask).astype(points.dtype)
        points[np.where(points[:, :, 2] > 2.0)] = [0, 0, 0]
        # remove background and container from label mask
        mask_img[np.where(mask_img < 4)] = 0

        return rgb_img, mask_img, points

    def process_file(self, args) -> None:
        hdf5_file, img_idx = args
        rgb_img, mask_img, points = self.__get_data_from_hdf5(hdf5_file)
        depth = points[:, :, 2]

        Image.fromarray(rgb_img).save(self.dst_rgb_dir + "/{:08d}.png".format(img_idx))
        np.save(self.dst_depth_dir + "/{:08d}.npy".format(img_idx), points)
        Image.fromarray(mask_img).save(
            self.dst_mask_dir + "/{:08d}.png".format(img_idx)
        )

        visualize_depth(
            depth,
            show=False,
            saveFile=self.dst_depth_img_dir + "/{:08d}.jpg".format(img_idx),
            showColorbar=False,
        )

        visualize_imgs(
            rgb_img,
            depth,
            mask_img,
            show=False,
            saveFile=self.dst_visualization_dir + "/{:08d}.jpg".format(img_idx),
        )

        return img_idx

    def generate(self):

        if self.visualize:
            # process one by one
            pbar = tqdm(self.all_files)
            for img_idx, hdf5_file in enumerate(pbar):

                self.process_file((hdf5_file, img_idx))

                rgb_img, mask_img, points = self.__get_data_from_hdf5(hdf5_file)

                visualize_imgs(rgb_img, points[:, :, 2], mask_img, show=True)
                # visualize_depth(points[:, :, 2], show=True, showColorbar=True)

                pbar.set_description("{}".format(img_idx))
                pbar.refresh()
        else:

            dst_file_indices = []
            num_cpus = cpu_count()
            # num_cpus = 8
            with get_context("spawn").Pool(processes=num_cpus) as pool:
                for idx in tqdm(
                    pool.imap(
                        self.process_file,
                        [(f, i) for i, f in enumerate(self.all_files)],
                    ),
                    total=len(self.all_files),
                ):
                    dst_file_indices.append(idx)

            if len(dst_file_indices) != len(self.all_files):
                print(
                    "missed {} files".format(
                        len(dst_file_indices) - len(self.all_files)
                    )
                )

            # create splits
            print("creating {} split...".format(self.split))
            num_indices = len(dst_file_indices)
            dst_file_indices = np.array(dst_file_indices)
            rand_indices = np.random.permutation(num_indices)
            rand_dst_file_indices = dst_file_indices[rand_indices]

            np.save(self.dst + "/{}.npy".format(self.split), rand_dst_file_indices)

            shutil.copyfile(
                os.path.join(self.src, "camera.json"),
                os.path.join(self.dst, "camera_params.json"),
            )


if __name__ == "__main__":
    set_start_method("spawn")
    args = parse_args()

    generator = Generator(args.src, args.dst, args.split, args.visualize)
    generator.generate()
