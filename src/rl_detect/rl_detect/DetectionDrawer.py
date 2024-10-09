import cv2
import numpy as np
import open3d as o3d
import pdb
import time
import os
from scipy.stats import zscore
class DetectionDrawer():

    def __init__(self, sam, class_names, pinhole_camera_intrinsic, extrinsic, intr):
        self.sam_model = sam
        self.class_names = class_names
        num_classes = len(class_names)
        self.pinhole_camera_intrinsic = pinhole_camera_intrinsic
        self.extrinsic = extrinsic
        self.intr = intr
        self.colors = self.generate_colors(num_classes)
        self.flip_matrix = np.array([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    def generate_colors(self, num_classes):
        colors = []
        for _ in range(num_classes):
            color = np.random.rand(3) * 255 
            if np.mean(color) < 128: 
                color = color + (255 - np.mean(color)) * 0.5
            colors.append(color)
        return np.array(colors)
    def __call__(self, image, depth_image, boxes, scores, class_ids):
        return self.draw_detections(image, depth_image, boxes, scores, class_ids)

    def bool_mask_to_int(self, mask, true_value=(255, 0, 0)):
        true_value = np.array(true_value)
        return mask.astype(np.uint8)[:, :, None] * true_value
    def draw_detections(self, image, depth_image, boxes, scores, class_ids):
        if class_ids.shape[0] == 0:
            return image
        bbox_3d = []       
        colors = self.colors[class_ids]
        results = self.sam_model(image, bboxes=boxes)
        self.i = 0
        pcl = o3d.geometry.PointCloud()
        mask_img = np.zeros_like(image)
        for color in colors:
            mask = results[0].masks[self.i].data.cpu().numpy()
            mask = mask.squeeze(axis=0).astype(np.uint8)
            bx3d, depth, pcd = self.draw_3d_bounding_box(depth_image, mask)
            normalized_color = [c / 255.0 for c in color]
            pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array(normalized_color), (pcd.points.__len__(), 1)))
            pcl.colors.extend(pcd.colors)
            pcl.points.extend(pcd.points)
            depth = depth * 0.001
            self.i += 1
            bbox_3d.append(bx3d)
        return bbox_3d, pcl

    def draw_3d_bounding_box(self, depth_image, color_mask):
        pcd = o3d.geometry.PointCloud()
        mask = color_mask
        
        eroded_ann_mask = cv2.erode(mask, kernel=np.ones((3, 3), np.uint8), iterations=1)
        isolated_depth = np.where((eroded_ann_mask > 0), depth_image, np.nan)
        depth = np.median(isolated_depth[~np.isnan(isolated_depth)])
        non_nan_points = np.argwhere(~np.isnan(isolated_depth))
        non_nan_depth_values = isolated_depth[non_nan_points[:, 0], non_nan_points[:, 1]]
        
        z_scores = zscore(non_nan_depth_values)
        # Filter out points with z-score greater than 2
        filtered_indices = np.abs(z_scores) <= 2.5
        non_nan_depth_values = non_nan_depth_values[filtered_indices]
        non_nan_points = non_nan_points[filtered_indices]

        # Intrinsics
        fx = self.intr.fx
        fy = self.intr.fy
        cx = self.intr.ppx
        cy = self.intr.ppy

        # Convert depth image to 3D point cloud
        u = non_nan_points[:, 1]  # x coordinates in image
        v = non_nan_points[:, 0]  # y coordinates in image
        z = non_nan_depth_values * 0.001 

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack((x, y, z), axis=-1)
        pcd.points = o3d.utility.Vector3dVector(points)        
        final_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=5,
                                                         std_ratio=2.0)
        # bbox_3d = denoised_pcd.get_oriented_bounding_box()
        bbox_3d = final_pcd.get_axis_aligned_bounding_box()
        bbox_3d.color = (0,0,1)
        return bbox_3d, depth, final_pcd
