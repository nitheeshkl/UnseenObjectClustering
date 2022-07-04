#!/usr/bin/env python3

import numpy as np
import cv2
import open3d as o3d
import json
import matplotlib.pyplot as plt

def compute_xyz(depth, camera_params):
    height = depth.shape[0]
    width = depth.shape[1]
    fx = camera_params['fx']
    fy = camera_params['fy']
    if "x_offset" in camera_params.keys():
        px = camera_params['x_offset']
        py = camera_params['y_offset']
    else:
        px = camera_params['cx']
        py = camera_params['cy']

    indices =  np.indices((height, width), dtype=np.float32).transpose(1,2,0) #[H,W,2]

    z = depth
    x = (indices[..., 1] - px) * z / fx
    y = (indices[..., 0] - py) * z / fy
    xyz_img = np.stack([x,y,z], axis=-1) # [H,W,3]

    return xyz_img

def visualize_xyz(xyz):
    # convert [H,W,3] to [N,3] required by o3d
    x = xyz[:,:,0].flatten()
    y = xyz[:,:,1].flatten()
    z = xyz[:,:,2].flatten()
    points = np.stack([x,y,z]).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0.,0.,0.])
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



def visualize_features(features):
    C,H,W = features.shape
    feature_img = np.zeros([H,W,3])
    for i in range(3):
        feature_img[:,:,i] = np.sum(features[i::3,:,:], axis=0)
    feature_img = normalize(feature_img)
    feature_img *= 255
    feature_img = feature_img.astype(np.uint8)
    plt.imshow(feature_img)
    plt.show()
