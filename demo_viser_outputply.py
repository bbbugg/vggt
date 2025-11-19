# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import time
import threading
import argparse
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2
import open3d as o3d

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def apply_sky_segmentation(conf: np.ndarray, image_folder: str) -> np.ndarray:
    """
    Apply sky segmentation to confidence scores.

    Args:
        conf (np.ndarray): Confidence scores with shape (S, H, W)
        image_folder (str): Path to the folder containing input images

    Returns:
        np.ndarray: Updated confidence scores with sky regions masked out
    """
    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    # Download skyseg.onnx if it doesn't exist
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    sky_mask_list = []

    print("Generating sky masks...")
    for i, image_path in enumerate(tqdm(image_files[:S])):  # Limit to the number of images in the batch
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

        # Resize mask to match H×W if needed
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        sky_mask_list.append(sky_mask)

    # Convert list to numpy array with shape S×H×W
    sky_mask_array = np.array(sky_mask_list)
    # Apply sky mask to confidence scores
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    conf = conf * sky_mask_binary

    print("Sky segmentation applied successfully")
    return conf


def save_points_to_ply(
        points: np.ndarray,
        colors: np.ndarray,
        confidences: np.ndarray,
        threshold: float,
        output_dir: str
) -> None:
    """
    保存置信度高于阈值的点云到PLY文件。
    
    Args:
        points (np.ndarray): 所有点坐标，shape (N, 3)
        colors (np.ndarray): 每个点的颜色，shape (N, 3)，范围 [0, 255], dtype=np.uint8
        confidences (np.ndarray): 每个点的置信度，shape (N,)
        threshold (float): 置信度阈值，只保留大于该值的点
        output_dir (str): 输出文件夹路径，会在此目录下生成 'point_cloud.ply'

    Returns:
        None
    """
    # 过滤置信度高的点
    valid_mask = confidences > threshold
    valid_points = points[valid_mask]
    valid_colors = colors[valid_mask]

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 构建完整输出路径
    output_path = os.path.join(output_dir, "arm_point_cloud.ply")

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.colors = o3d.utility.Vector3dVector(valid_colors / 255.0)  # 归一化到 [0,1]

    # 保存为 PLY 文件
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved filtered point cloud to {output_path}")


parser = argparse.ArgumentParser(description="VGGT demo with viser for 3D visualization")
parser.add_argument(
    "--image_folder", type=str, default="input_images/", help="Path to folder containing images"
)
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
parser.add_argument(
    "--conf_threshold", type=float, default=0.50, help="Initial percentage of low-confidence points to filter out"
)
parser.add_argument("--mask_sky", action="store_true", default=False,
                    help="Apply sky segmentation to filter out sky points")
parser.add_argument("--output_dir", default="input_images/", type=str,
                    help="Path to save the output PLY file (e.g., 'output.ply').")


def main():
    """
    Main function for the VGGT demo with viser for 3D visualization.

    This function:
    1. Loads the VGGT model
    2. Processes input images from the specified folder
    3. Runs inference to generate 3D points and camera poses
    4. Optionally applies sky segmentation to filter out sky points
    5. Visualizes the results using viser

    Command-line arguments:
        --image_folder: Path to folder containing input images  # 输入图片文件夹路径
        --use_point_map: Use point map instead of depth-based points  # 使用点云图而不是基于深度的点
        --background_mode: Run the viser server in background mode  # 以后台模式运行viser服务器
        --port: Port number for the viser server  # viser服务器端口号
        --conf_threshold: Initial percentage of low-confidence points to filter out  # 初始置信度阈值，过滤低置信度点
        --mask_sky: Apply sky segmentation to filter out sky points  # 应用天空分割，过滤天空点
    """
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing and loading VGGT model...")

    model = VGGT()
    print("begin to load state_dict...")
    model.load_state_dict(torch.load("ckpt/model.pt"))
    print("successfully loading!")

    model.eval()
    model = model.to(device)

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = glob.glob(os.path.join(args.image_folder, "*"))
    print(f"Found {len(image_names)} images")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    print("Processing model outputs...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

    if args.use_point_map:
        print("Using 3D points from point map")
    else:
        print("Using 3D points by unprojecting depth map by cameras")

    if args.mask_sky:
        print("Sky segmentation enabled - will filter out sky points")

    # Unpack prediction dict
    images = predictions["images"]  # (S, 3, H, W)
    world_points_map = predictions["world_points"]  # (S, H, W, 3)
    conf_map = predictions["world_points_conf"]  # (S, H, W)

    depth_map = predictions["depth"]  # (S, H, W, 1)
    depth_conf = predictions["depth_conf"]  # (S, H, W)

    extrinsics_cam = predictions["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = predictions["intrinsic"]  # (S, 3, 3)

    # Compute world points from depth if not using the precomputed point map 如果不使用预计算的点图，则从深度计算世界点
    if not args.use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    # Apply sky segmentation if enabled
    if args.mask_sky and args.image_folder is not None:
        conf = apply_sky_segmentation(conf, args.image_folder)

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    save_points_to_ply(points=points, colors=colors_flat, confidences=conf_flat, threshold=args.conf_threshold,
                       output_dir=args.output_dir)

    # cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically
    # # For convenience, we store only (3,4) portion
    # cam_to_world = cam_to_world_mat[:, :3, :]

    # # Compute scene center and recenter
    # scene_center = np.mean(points, axis=0)
    # points_centered = points - scene_center
    # cam_to_world[..., -1] -= scene_center

    # # Store frame indices so we can filter by frame
    # frame_indices = np.repeat(np.arange(S), H * W)

    print("Visualization complete")


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=7 python demo_viser_outputply.py --image_folder /mnt/workspace/xiaoyu/vggt/examples/vil_test --output_dir /mnt/workspace/xiaoyu/vggt/examples/vil_test_output --mask_sky 2>&1 | tee /mnt/workspace/xiaoyu/crossview_localization_v1/vggt_output.log

# CUDA_VISIBLE_DEVICES=7 python demo_viser_outputply.py --image_folder /mnt/workspace/xiaoyu/vggt/examples/vil_test --output_dir /mnt/workspace/xiaoyu/vggt/examples/vil_test_output 2>&1 | tee /mnt/workspace/xiaoyu/crossview_localization_v1/vggt_output.log
