import cv2
import numpy as np
import open3d as o3d

class DetectionDrawer():

    def __init__(self, sam, class_names, pinhole_camera_intrinsic, extrinsic):
        self.sam_model = sam
        self.class_names = class_names
        self.pinhole_camera_intrinsic = pinhole_camera_intrinsic
        self.extrinsic = extrinsic
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
        img_height, img_width = image.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)
        
        classes = self.class_names[class_ids]
        colors = self.colors[class_ids]
        results = self.sam_model.predict(image, bboxes=boxes)[0]
        i = 0
        for box, score, label, color in zip(boxes, scores, classes, colors):
            x1, y1, x2, y2 = box.astype(int)
            object_depth = np.median(depth_image[int(y1):int(y2), int(x1):int(x2)])
            # mask_img = np.zeros_like(image)
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
            mask = results[i].masks.data.cpu().numpy()
            i += 1
            mask = mask.squeeze(axis=0)
            #remove outliers in mask
            color_mask = self.bool_mask_to_int(mask, true_value=color)
            color_mask = color_mask.astype(image.dtype)
            mask_img = cv2.add(mask_img, color_mask)
            caption = f'{label} {score*100:.2f}%' + f' {object_depth:.2f}m'
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                          fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)

            cv2.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
            cv2.putText(det_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

            bx3d, pcd = self.draw_3d_bounding_box(image, depth_image, box, mask)
            pcl.points.extend(pcd.points)
            bbox_3d.append(bx3d)
        o3d.io.write_point_cloud("/home/rameez/pointclouds.ply", pcl)
        o3d.visualization.draw_geometries([pcl] + bbox_3d)
        return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0), mask_img

    def get_colors(self, num_classes):
        return self.rng.uniform(0, 255, size=(num_classes, 3)) #use predefined colors

    def draw_3d_bounding_box(self, color_image, depth_image, bbox, color_mask):
        pcd = o3d.geometry.PointCloud()

        mask = np.array(color_mask).astype(np.uint8)
        
        # Erode the annotation mask (to avoid reconstructing in 3D some background)
        eroded_ann_mask = cv2.erode(mask, kernel=np.ones((3, 3), np.uint8), iterations=1)
        isolated_depth = np.where((eroded_ann_mask > 0) & (depth_image < 1000), depth_image, np.nan)
        non_nan_points = np.argwhere(~np.isnan(isolated_depth))
        non_nan_depth_values = isolated_depth[non_nan_points[:, 0], non_nan_points[:, 1]]
        # depth_scale = 1000.0

        pcd.points = o3d.utility.Vector3dVector(
            np.column_stack([non_nan_points[:, 1], non_nan_points[:, 0], non_nan_depth_values])
        )        
        # o3d.visualization.draw_geometries([pcd])
        # pcd_outlier = pcd.voxel_down_sample(voxel_size=0.01)

        denoised_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=300,
                                                                std_ratio=2.0)
        denoised_pcd.transform(self.flip_matrix)

        # o3d.visualization.draw_geometries([denoised_pcd])
        bbox_3d = denoised_pcd.get_axis_aligned_bounding_box()        
        random_color = self.rng.uniform(0, 1, size=(3,))
        bbox_3d.color = random_color
        bbox_3d.color = (0,0,1)

        return bbox_3d, denoised_pcd
