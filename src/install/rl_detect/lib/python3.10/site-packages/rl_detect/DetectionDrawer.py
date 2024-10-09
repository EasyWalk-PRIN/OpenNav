import cv2
import numpy as np
import open3d as o3d
import pdb
import time
class DetectionDrawer():

    def __init__(self, sam, class_names, pinhole_camera_intrinsic, extrinsic, intr):
        self.sam_model = sam
        self.class_names = class_names
        self.pinhole_camera_intrinsic = pinhole_camera_intrinsic
        self.extrinsic = extrinsic
        self.intr = intr
        self.colors = np.array([
                                [ 21.84053762,  60.38667918, 204.32498863],  # Class 1
                                [148.4513192 ,  24.00280377, 110.44736976],  # Class 2
                                [122.15808103,  40.73342323, 187.31717361],  # Class 3
                                [ 28.98636508,  99.76318858, 131.76874657],  # Class 4
                                [109.81014521, 149.63363572, 188.14863576],  # Class 5
                                [243.84814998,  72.47129676, 165.37953781],  # Class 6
                                [177.53507915,  74.643791  ,   0.37997129],  # Class 7
                                [248.23237007,  76.09231187,  80.06643052],  # Class 8
                                [227.38632296, 149.21654967, 120.18396462],  # Class 9
                                [197.18563746,   7.73823195, 180.27609939],  # Class 10
                                [ 95.43217754,  23.16744194, 168.42751719]   # Class 11
                            ])
        self.flip_matrix = np.array([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    def __call__(self, image, depth_image, boxes, scores, class_ids, mask_alpha=0.3):
        return self.draw_detections(image, depth_image, boxes, scores, class_ids, mask_alpha)

    def bool_mask_to_int(self, mask, true_value=(255, 0, 0)):
        true_value = np.array(true_value)
        return mask.astype(np.uint8)[:, :, None] * true_value

    def draw_detections(self, image, depth_image, boxes, scores, class_ids, mask_alpha=0.3):
        if class_ids.shape[0] == 0:
            return image
        bbox_3d = []
        # detections = {
        #     "class_name": [],
        #     "score": [],
        #     "x": [],
        #     "y": [],
        #     "z": [],
        #     "depth": []
        # }        
        classes = self.class_names[class_ids]
        colors = self.colors[class_ids]
        # start_time = time.time()
        results = self.sam_model.predict(image, bboxes=boxes)[0]
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Time taken to predict the mask SAM: {elapsed_time}")
        i = 0
        # start_time = time.time()
        for box, score, label, color in zip(boxes, scores, classes, colors):
            mask = results[i].masks.data.cpu().numpy()
            i += 1
            mask = mask.squeeze(axis=0).astype(np.uint8)
            # color_mask = self.bool_mask_to_int(mask, true_value=color)
            # color_mask = color_mask.astype(image.dtype)

            bx3d, depth = self.draw_3d_bounding_box(depth_image, mask)
            depth = depth * 0.001
            
            # Compute dimensions of the bounding box
            # bbox_dims = bx3d.get_extent()  # This returns a 3-tuple (width, height, depth)
            # bbox_dims = bx3d.extent # This returns a 3-tuple (width, height, depth)
            bbox_3d.append(bx3d)
        #     detections["class_name"].append(label)
        #     detections["score"].append(score*100)
        #     detections["x"].append(bbox_dims[0])
        #     detections["y"].append(bbox_dims[1])
        #     detections["z"].append(bbox_dims[2])
        #     detections["depth"].append(depth)
        # print(detections)
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Time taken to draw the 3D bounding boxes: {elapsed_time}")

        return bbox_3d

    def draw_3d_bounding_box(self, depth_image, color_mask):
        pcd = o3d.geometry.PointCloud()
        mask = color_mask
        
        # Erode the annotation mask (to avoid reconstructing in 3D some background)
        eroded_ann_mask = cv2.erode(mask, kernel=np.ones((3, 3), np.uint8), iterations=1)
        isolated_depth = np.where((eroded_ann_mask > 0), depth_image, np.nan)
        depth = np.median(isolated_depth[~np.isnan(isolated_depth)])
        non_nan_points = np.argwhere(~np.isnan(isolated_depth))
        non_nan_depth_values = isolated_depth[non_nan_points[:, 0], non_nan_points[:, 1]]
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

        points = np.stack((x, y, z), axis=-1)  # Create an (N, 3) array

        pcd.points = o3d.utility.Vector3dVector(points)        
        # pcd = pcd.uniform_down_sample(every_k_points=10)
        # start_time = time.time()
        denoised_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50,
                                                         std_ratio=2.0)
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Time taken to denoise the point cloud: {elapsed_time}")
        # denoised_pcd.transform(self.flip_matrix)

        # bbox_3d = denoised_pcd.get_oriented_bounding_box()
        # start_time = time.time() 
        bbox_3d = denoised_pcd.get_axis_aligned_bounding_box()
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Time taken to get the axis aligned bounding box: {elapsed_time}")
        bbox_3d.color = (0,0,1)
        return bbox_3d, depth
