# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

import os, sys
from pathlib import Path

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)


def read_trajectory(extrinsics_dir):
    traj = []
    log_files = sorted([f for f in os.listdir(extrinsics_dir) if f.endswith('.log')])
    for file_name in log_files:
        file_path = os.path.join(extrinsics_dir, file_name)
        if os.path.isfile(file_path):
            try:
                data = np.loadtxt(file_path) 
                traj.append(data)
            except Exception as e:
                print(f"Error loading file {file_name}: {e}")
    
    return traj


def rotation_matrix(axis, angle):
    """
    Compute a 3D rotation matrix given an axis (X, Y, or Z) and an angle in degrees.
    """
    angle_rad = np.radians(angle)
    if axis == 'X':
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad), 0],
            [0, np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'Y':
        return np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'Z':
        return np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0, 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Please choose 'X', 'Y', or 'Z'.")


def translation_matrix(axis, translation):
    """
    Compute a 3D translation matrix given an axis (X, Y, or Z) and a translation in meters.
    """
    if axis == 'X':
        return np.array([
            [1, 0, 0, translation],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'Y':
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, translation],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'Z':
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, translation],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Please choose 'X', 'Y', or 'Z'.")


if __name__ == "__main__":
    voxel_length = 0.004
    sdf_trunc = 0.02
    depth_max = 6
    data_dir = Path("/home/cra80/Projects/threestudio-sketch/outputs/mvdream+sd-increased_weights/teapot_001@20240409-035530/save/open3d")

    color_dir = data_dir / "color"
    depth_dir = data_dir / "depth"
    intrinsics_path = data_dir / "intrinsics.json"
    extrinsics_dir = data_dir / "extrinsics"

    camera_poses = read_trajectory(extrinsics_dir)
    camera_intrinsics =  o3d.io.read_pinhole_camera_intrinsic(str(intrinsics_path))

    color_paths = sorted([str(color_dir / f) for f in os.listdir(color_dir) if f.endswith('.png')])
    depth_paths = sorted([str(depth_dir / f) for f in os.listdir(depth_dir) if f.endswith('.png')])


    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    pcds = []

    distance = 3.0

    translation = translation_matrix(axis="Z", translation=-distance)
    rotations = np.arange(0, 360, 3, dtype=np.int32)

    for i in range(len(camera_poses)):
        print("Integrate {:d}-th image into the volume.".format(i))
        color = o3d.io.read_image(color_paths[i])
        depth = o3d.io.read_image(depth_paths[i])

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=depth_max, convert_rgb_to_intensity=False)
        

        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                camera_intrinsics)
        #transform = np.linalg.inv(camera_poses[i])

        rotation = rotation_matrix(angle=-1*rotations[i], axis="Y")
        transform = rotation @ translation
        pcd.transform(transform)
        # Flip it, otherwise the pointcloud will be upside down
        #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcds.append(pcd)
                
        # plt.subplot(1, 2, 1)
        # plt.title('image')
        # plt.imshow(rgbd.color)
        # plt.subplot(1, 2, 2)
        # plt.title('depth image')
        # plt.imshow(rgbd.depth)
        # plt.show()

        volume.integrate(
            rgbd,
            camera_intrinsics,
            np.linalg.inv(transform),
        )

     # Step 1 - Get scene objects
    meshFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    print("Visualize depth point clouds")
    o3d.visualization.draw_geometries(pcds + [meshFrame,])

    print("Extract triangle mesh")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([mesh])

    print("Extract voxel-aligned debugging point cloud")
    voxel_pcd = volume.extract_voxel_point_cloud()
    voxel_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([voxel_pcd])