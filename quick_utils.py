#!/usr/bin/env python3
import os
import glob
import numpy as np
import cv2
import open3d as o3d
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch

def compute_xyz(depth, camera_params, scaled=False):
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
        
    print("scale = ({},{})".format(scale_x, scale_y))

    fx, fy = fx * scale_x, fy * scale_y
    px, py = px * scale_x, py * scale_y

    z = depth
    x = (indices[..., 1] - px) * z / fx
    y = (indices[..., 0] - py) * z / fy
    xyz_img = np.stack([x,y,z], axis=-1) # [H,W,3]

    return xyz_img

def visualize_xyz(xyz, origin=[0.,0.,0.]):
    # convert [H,W,3] to [N,3] required by o3d
    # x = xyz[:,:,0].flatten()
    # y = xyz[:,:,1].flatten()
    # z = xyz[:,:,2].flatten()
    # points = np.stack([x,y,z]).T
    points = xyz.reshape(-1,3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=origin)
    geometries = list([pcd, coordinate_frame])
    
    o3d.visualization.draw_geometries(geometries)

    # viewer = o3d.visualization.Visualizer()
    # viewer.create_window()
    # for geometry in geometries:
    #     viewer.add_geometry(geometry)
    # opt = viewer.get_render_option()
    # opt.show_coordinate_frame = True
    # opt.background_color = np.asarray([0.5, 0.5, 0.5])
    # viewer.run()
    # viewer.destroy_window()

def xyz_to_o3d_pcd(xyz):
    # convert [H,W,3] to [N,3] required by o3d
    x = xyz[:,:,0].flatten()
    y = xyz[:,:,1].flatten()
    z = xyz[:,:,2].flatten()
    points = np.stack([x,y,z]).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

def visualize_depth(depth, camera_params):
    xyz = compute_xyz(depth, camera_params)

    visualize_xyz(xyz)

def load_depth(img_filename):
    depth_img = cv2.imread(img_filename, cv2.IMREAD_ANYDEPTH)
    depth = depth_img.astype(np.float32) / 1000.0

    return depth

def load_camera_params(params_filename):
    params = None
    with open(params_filename) as f:
        params = json.load(f)

    return params

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

def visualize_features_list(features_list, return_img=False):
    num_feats = len(features_list)

    fig, axs = plt.subplots(1,num_feats, figsize=[35,4])
    canvas = FigureCanvas(fig)


    for i, f in enumerate(features_list):
        feature = torch.load(f)
        feature_img = feature_tensor_to_img(feature)
        axs[i].imshow(feature_img)
        axs[i].set_title(os.path.splitext(os.path.basename(f))[0])

    fig.tight_layout()
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')

    if return_img:
        return image
    else:
        plt.show()
        return None


def visualize_feature_tensors_dir(dirpath, return_img = False):
    features_list = sorted(glob.glob(os.path.join(dirpath, '*.pt')))
    num_feats = len(features_list)

    fig, axs = plt.subplots(1,num_feats, figsize=[35,4])
    canvas = FigureCanvas(fig)


    for i, f in enumerate(features_list):
        feature = torch.load(f)
        feature_img = feature_tensor_to_img(feature)
        axs[i].imshow(feature_img)
        axs[i].set_title(os.path.splitext(os.path.basename(f))[0])

    fig.tight_layout()
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')

    if return_img:
        return image
    else:
        plt.show()
        return None


def features_to_img(features):
    C,H,W = features.shape
    feature_img = np.zeros([H,W,3])
    for i in range(3):
        feature_img[:,:,i] = np.sum(features[i::3,:,:], axis=0)
    feature_img = normalize(feature_img)
    feature_img *= 255
    feature_img = feature_img.astype(np.uint8)
    return feature_img

def visualize_features(features):
    features_img = features_to_img(features)
    plt.imshow(features_img)
    plt.show()

def visualize_features_dir(dirpath):
    features_list = sorted(glob.glob(os.path.join(dirpath, '*.npy')))
    num_feats = len(features_list)

    fig, axs = plt.subplots(1,num_feats, figsize=[40,10])

    for i, f in enumerate(features_list):
        feature = np.load(f)
        feature = feature[0]
        feature_img = features_to_img(feature)
        axs[i].imshow(feature_img)
        axs[i].set_title(os.path.splitext(os.path.basename(f))[0])

    fig.tight_layout()
    plt.show()
