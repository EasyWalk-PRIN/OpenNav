import cv2
import numpy as np
import open3d as o3d

class DetectionDrawer():

    def __init__(self, sam, class_names, pinhole_camera_intrinsic, extrinsic, intr):
        self.sam_model = sam
        self.class_names = class_names
        self.pinhole_camera_intrinsic = pinhole_camera_intrinsic
        self.extrinsic = extrinsic
        self.intr = intr
        self.rng = np.random.default_rng(3)
        self.colors = self.get_colors(len(class_names))
        self.flip_matrix = np.array([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    def __call__(self, image, depth_image, boxes, scores, class_ids, mask_alpha=0.3):
        return self.draw_detections(image, depth_image, boxes, scores, class_ids, mask_alpha)

    def update_class_names(self, class_names):
        for i, class_name in enumerate(class_names):
            if class_name not in self.class_names:
                self.colors[i] = self.get_colors(1)
            else:
                self.colors[i] = self.colors[list(self.class_names).index(class_name)]
        self.class_names = class_names

    def bool_mask_to_int(self, mask, true_value=(255, 0, 0)):
        true_value = np.array(true_value)
        return mask.astype(np.uint8)[:, :, None] * true_value

    def draw_detections(self, image, depth_image, boxes, scores, class_ids, mask_alpha=0.3):
        if class_ids.shape[0] == 0:
            return image
        mask_img = np.zeros_like(image)  # Initialize mask image
        det_img = image.copy()
        pcl = o3d.geometry.PointCloud()
        bbox_3d = []
        detections = {
            "class_name": [],
            "score": [],
            "x": [],
            "y": [],
            "z": [],
            "depth": []
        }
        img_height, img_width = image.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)
        
        classes = self.class_names[class_ids]
        colors = self.colors[class_ids]
        results = self.sam_model.predict(image, bboxes=boxes)[0]
        i = 0
        for box, score, label, color in zip(boxes, scores, classes, colors):
            x1, y1, x2, y2 = box.astype(int)
            object_depth = np.median(depth_image[int(y1):int(y2), int(x1):int(x2)]) / 100.0
            # mask_img = np.zeros_like(image)
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
            mask = results[i].masks.data.cpu().numpy()
            i += 1
            mask = mask.squeeze(axis=0).astype(np.uint8)
            #remove outliers in mask
            color_mask = self.bool_mask_to_int(mask, true_value=color)
            color_mask = color_mask.astype(image.dtype)
            mask_img = cv2.add(mask_img, color_mask)
            caption = f'{label} {score*100:.2f}%' 
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                          fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)

            cv2.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
            cv2.putText(det_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

            bx3d, pcd, depth = self.draw_3d_bounding_box(image, depth_image, box, mask)
            depth = depth * 0.01
            
            # Compute dimensions of the bounding box
            # bbox_dims = bx3d.get_extent()  # This returns a 3-tuple (width, height, depth)
            bbox_dims = bx3d.extent # This returns a 3-tuple (width, height, depth)
            # import pdb
            # pdb.set_trace()
            dimensions_caption = f'X: {bbox_dims[0]:.2f} Y: {bbox_dims[1]:.2f} Z: {bbox_dims[2]:.2f}m Depth: {depth:.2f}m'
            caption += f' {dimensions_caption}'

            # Display dimensions
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                          fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)
            cv2.rectangle(det_img, (x1, y1 - th), (x1 + tw, y1 - 2 * th), color, -1)
            cv2.putText(det_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)
            # print(caption)
            normalized_color = [c / 255.0 for c in color]
            pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array(normalized_color), (pcd.points.__len__(), 1)))
            pcl.colors.extend(pcd.colors)
            pcl.points.extend(pcd.points)
            bbox_3d.append(bx3d)
            detections["class_name"].append(label)
            detections["score"].append(score*100)
            detections["x"].append(bbox_dims[0])
            detections["y"].append(bbox_dims[1])
            detections["z"].append(bbox_dims[2])
            detections["depth"].append(depth)
        print(detections)
        # o3d.io.write_point_cloud("/home/rameez/pointclouds.ply", pcl)
        # import pdb
        # pdb.set_trace()
        o3d_color = o3d.geometry.Image(image.astype(np.uint8))
        o3d_depth = o3d.geometry.Image(depth_image.astype(np.float32))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale= 1000.0, convert_rgb_to_intensity=True)

        pcd_orig = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.pinhole_camera_intrinsic, self.extrinsic)
        #save point cloud
        
        o3d.io.write_point_cloud("/home/rameez/pointcloudo3d.ply", pcd_orig)
        # o3d.visualization.draw_geometries([pcd_orig])                
        # o3d.visualization.draw_geometries([pcd_orig] + bbox_3d)        
        # o3d.visualization.draw_geometries([pcl] + bbox_3d)
        pcd_orig.transform(self.flip_matrix)
        return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0), mask_img , pcd_orig, bbox_3d

    def get_colors(self, num_classes):
        return self.rng.uniform(0, 255, size=(num_classes, 3))

    def draw_3d_bounding_box(self, color_image, depth_image, bbox, color_mask):
        pcd = o3d.geometry.PointCloud()
        import pdb
        # pdb.set_trace()
        mask = color_mask
        
        # Erode the annotation mask (to avoid reconstructing in 3D some background)
        eroded_ann_mask = cv2.erode(mask, kernel=np.ones((3, 3), np.uint8), iterations=1)
        isolated_depth = np.where((eroded_ann_mask > 0), depth_image, np.nan)
        depth = np.median(isolated_depth[~np.isnan(isolated_depth)])
        non_nan_points = np.argwhere(~np.isnan(isolated_depth))
        non_nan_depth_values = isolated_depth[non_nan_points[:, 0], non_nan_points[:, 1]]
        # depth_scale = 0.001
        # z = non_nan_depth_values* depth_scale
        # Intrinsics
        fx = self.intr.fx
        fy = self.intr.fy
        cx = self.intr.ppx
        cy = self.intr.ppy

        # Convert depth image to 3D point cloud
        points = []
        for i in range(non_nan_points.shape[0]):
            u = non_nan_points[i, 1]  # x coordinate in image
            v = non_nan_points[i, 0]  # y coordinate in image
            z = non_nan_depth_values[i] * 0.001  # Convert depth to meters
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])
        pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.points = o3d.utility.Vector3dVector(
        #     np.column_stack([non_nan_points[:, 1] / 100.0, non_nan_points[:, 0] / 100.0, non_nan_depth_values / 100.0])
        # )        
        denoised_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=300,
                                                         std_ratio=2.0)

        # denoised_pcd.transform(self.flip_matrix)
        # denoised_pcd.transform(self.extrinsic)

        # o3d.visualization.draw_geometries([denoised_pcd])
        bbox_3d = denoised_pcd.get_oriented_bounding_box() 
        # bbox_3d = denoised_pcd.get_axis_aligned_bounding_box()
        # pdb.set_trace()
        # pdb.set_trace() 
        random_color = self.rng.uniform(0, 1, size=(3,))
        bbox_3d.color = random_color
        bbox_3d.color = (0,0,1)
        # print(depth*0.01)
        return bbox_3d, denoised_pcd, depth
