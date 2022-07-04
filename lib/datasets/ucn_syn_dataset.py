from ctypes import util
import torch
import torch.utils.data as data
import os
import json
import cv2
import numpy as np
import imageio

import datasets
from fcn.config import cfg
from utils.blob import chromatic_transform, add_noise
from utils import augmentation
from utils import mask as util_

data_loading_params = {
    # Camera/Frustum parameters
    "img_width": 1920,
    "img_height": 1080,
    "near": 0.01,
    "far": 100,
    "fov": 45,  # vertical field of view in degrees
    "use_data_augmentation": True,
    # Multiplicative noise
    "gamma_shape": 1000.0,
    "gamma_scale": 0.001,
    # Additive noise
    "gaussian_scale": 0.005,  # 5mm standard dev
    "gp_rescale_factor": 4,
    # Random ellipse dropout
    "ellipse_dropout_mean": 10,
    "ellipse_gamma_shape": 5.0,
    "ellipse_gamma_scale": 1.0,
    # Random high gradient dropout
    "gradient_dropout_left_mean": 15,
    "gradient_dropout_alpha": 2.0,
    "gradient_dropout_beta": 5.0,
    # Random pixel dropout
    "pixel_dropout_alpha": 1.0,
    "pixel_dropout_beta": 10.0,
}


def compute_xyz(depth_img: np.ndarray, camera_params: dict) -> np.ndarray:
    """ Compute ordered point clouds from recorded depth image and camera intrinsics.
    """

    img_width = camera_params["width"]
    img_height = camera_params["height"]
    px, py = camera_params["cx"], camera_params["cy"]
    fx, fy = camera_params["fx"], camera_params["fy"]

    indices = np.indices((img_height, img_width), dtype=np.float32).transpose(
        1, 2, 0
    )  # [H,W,2]

    z = depth_img
    x = (indices[..., 1] - px) * z / fx
    y = (indices[..., 0] - py) * z / fy
    xyz_img = np.stack([x, y, z], axis=-1)  # [H,W,3]

    return xyz_img


class UcnSynDataset(data.Dataset, datasets.imdb):
    def __init__(self, image_set: str, ucn_syn_dataset_path: str = None):

        self._name = "ucn_syn_dataset_" + image_set
        self._image_set = image_set
        self._ucn_syn_dataset_path = (
            self._get_default_path()
            if ucn_syn_dataset_path is None
            else ucn_syn_dataset_path
        )
        self._classes = ("__background__", "foreground")
        self._pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()
        self.params = data_loading_params

        indices_file = os.path.join(
            self._ucn_syn_dataset_path, "{}.npy".format(self._image_set)
        )
        assert os.path.exists(indices_file), "{} does not exist".format(indices_file)
        self._indices = np.load(indices_file).tolist()
        self._size = len(self._indices)

        camera_params_file = os.path.join(
            self._ucn_syn_dataset_path, "camera_params.json"
        )
        assert os.path.exists(camera_params_file), "{} does not exist".format(
            camera_params_file
        )
        with open(camera_params_file) as f:
            self._camera_params = json.load(f)

    def __len__(self):
        return self._size

    def _get_default_path(self):
        return os.path.join(datasets.ROOT_DIR, "data", "ucn_syn")

    def __getitem__(self, index: int):

        file_index = self._indices[index]
        rgb_file = os.path.join(
            self._ucn_syn_dataset_path, "rgb", "{}.png".format(file_index)
        )
        depth_file = os.path.join(
            self._ucn_syn_dataset_path, "depth", "{}.png".format(file_index)
        )
        mask_file = os.path.join(
            self._ucn_syn_dataset_path, "mask", "{}.png".format(file_index)
        )

        # label
        label = np.asarray(imageio.imread(mask_file))
        # TODO: check if remap label is needed for DoPose_UCN, since the regenerated dataset has instance IDs from 1 - N
        # and background and table/container is already 0
        label = self.remap_label(label)

        # rgb image
        rgb_img = np.asarray(imageio.imread(rgb_file))
        # depth image
        if cfg.INPUT == "DEPTH" or cfg.INPUT == "RGBD":
            depth_img = np.asarray(imageio.imread(depth_file))
            xyz_img = self.process_depth(depth_img)
        else:
            xyz_img = None

        if cfg.TRAIN.SYN_CROP:
            rgb_img, xyz_img, label = self.pad_crop_resize(rgb_img, xyz_img, label)
            # FIXME: check if remap_label() is needed after SYN_CROP
            # label = self.remap_label(label)
        else:
            scale_percent = 40  # percent of original size
            width = int(rgb_img.shape[1] * scale_percent / 100)
            height = int(rgb_img.shape[0] * scale_percent / 100)
            dim = (width, height)

            rgb_img = cv2.resize(rgb_img, dim)
            xyz_img = cv2.resize(xyz_img, dim, cv2.INTER_NEAREST)
            label = cv2.resize(label.astype(np.uint8), dim, cv2.INTER_NEAREST)

        # sample label pixels
        if cfg.TRAIN.EMBEDDING_SAMPLING:
            label = self.sample_pixels(label, cfg.TRAIN.EMBEDDING_SAMPLING_NUM)

        if cfg.TRAIN.CHROMATIC and cfg.MODE == "TRAIN" and np.random.rand(1) > 0.1:
            rgb_img = chromatic_transform(rgb_img)
        if cfg.TRAIN.ADD_NOISE and cfg.MODE == "TRAIN" and np.random.rand(1) > 0.1:
            rgb_img = add_noise(rgb_img)

        rgb_tensor = torch.from_numpy(rgb_img) / 255.0
        rgb_tensor -= self._pixel_mean
        rgb_tensor = rgb_tensor.permute(2, 0, 1)
        label_tensor = torch.from_numpy(label).unsqueeze(0)

        item = {
            "label": label_tensor,
            "image_color": rgb_tensor,
        }

        if cfg.INPUT == "DEPTH" or cfg.INPUT == "RGBD":
            xyz_tensor = torch.from_numpy(xyz_img).permute(2, 0, 1)
            item["depth"] = xyz_tensor

        return item

    def remap_label(self, label):
        unique_indices = np.unique(label)
        remapped_label = label.copy()
        for k in range(unique_indices.shape[0]):
            remapped_label[label == unique_indices[k]] = k
        label = remapped_label
        return label

    def process_depth(self, depth_img: np.ndarray) -> np.ndarray:

        # convert millimeters -> meters
        depth_img = (depth_img / 1000.0).astype(np.float32)

        # add randome noise
        if self.params["use_data_augmentation"]:
            depth_img = augmentation.add_noise_to_depth(depth_img, self.params)
            depth_img = augmentation.dropout_random_ellipses(depth_img, self.params)

        # compute xyz point clouds
        xyz_img = compute_xyz(depth_img, self._camera_params)
        if self.params["use_data_augmentation"]:
            xyz_img = augmentation.add_noise_to_xyz(xyz_img, depth_img, self.params)

        return xyz_img

    def sample_pixels(self, label, num=1000):
        # -1 will be ignored
        label_new = -1 * np.ones_like(label)
        K = np.max(label)  # number of object instances
        for i in range(K + 1):
            index = np.where(label == i)
            n = len(index[0])  # num pixels belonging to instance i
            if n <= num:
                label_new[index[0], index[1]] = i
            else:
                perm = np.random.permutation(n)
                selected = perm[:num]
                label_new[index[0][selected], index[1][selected]] = i
        return label_new

    def pad_crop_resize(self, img, depth, mask):

        H, W, _ = img.shape

        K = np.max(mask)
        while True:
            if K > 0:
                idx = np.random.randint(1, K + 1)
            else:
                idx = 0
            foreground = (mask == idx).astype(np.float32)

            # get bbox around mask
            x_min, y_min, x_max, y_max = util_.mask_to_tight_box(foreground)
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2

            # make bbox square
            x_delta = x_max - x_min  # bbox width
            y_delta = y_max - y_min  # bbox height
            if x_delta > y_delta:  # width > height
                # increase height
                y_min = cy - x_delta / 2
                y_max = cy + x_delta / 2
            else:
                # increase width
                x_min = cx - y_delta / 2
                x_max = cx + y_delta / 2

            sidelen = x_max - x_min
            padding_percent = np.random.uniform(
                cfg.TRAIN.min_padding_percentage, cfg.TRAIN.max_padding_percentage
            )
            padding = int(round(sidelen * padding_percent))
            if padding == 0:
                padding = 25

            # Pad without affecting boundaries
            x_min = max(int(x_min - padding), 0)
            x_max = min(int(x_max + padding), W - 1)
            y_min = max(int(y_min - padding), 0)
            y_max = max(int(y_max + padding), H - 1)

            # FIXME: check if the patch is square after padding (if required)

            # crop
            if (y_min == y_max) or (x_min == x_max):  # if zero width or height
                continue  # continue to find next crop

            img_crop = img[y_min : y_max + 1, x_min : x_max + 1]
            mask_crop = mask[y_min : y_max + 1, x_min : x_max + 1]
            roi = [x_min, y_min, x_max, y_max]
            if depth is not None:
                depth_crop = depth[y_min : y_max + 1, x_min : x_max + 1]

            break  # exit loop since we found a required crop

        # resize
        s = cfg.TRAIN.SYN_CROP_SIZE
        img_crop = cv2.resize(img_crop, (s, s))
        mask_crop = cv2.resize(mask_crop, (s, s), interpolation=cv2.INTER_NEAREST)
        if depth is not None:
            depth_crop = cv2.resize(depth_crop, (s, s), interpolation=cv2.INTER_NEAREST)
        else:
            depth_crop = None

        return img_crop, depth_crop, mask_crop
