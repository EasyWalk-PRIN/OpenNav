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

            bx3d, pcd = self.draw_3d_bounding_box(image, depth_image, box, color_mask)
            pcl.points.extend(pcd.points)
            # pcl.colors.extend(pcd.colors)

            bbox_3d.append(bx3d)
        o3d.io.write_point_cloud("/home/rameez/pointclouds.ply", pcl)
        o3d.visualization.draw_geometries([pcl] + bbox_3d)
        return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0), mask_img

    def get_colors(self, num_classes):
        return self.rng.uniform(0, 255, size=(num_classes, 3))

    def draw_3d_bounding_box(self, color_image, depth_image, bbox, color_mask):
        # x1, y1, x2, y2 = bbox.astype(int)
        
        # # Ensure the bounding box coordinates are within image bounds
        # x1 = max(0, x1)
        # y1 = max(0, y1)
        # x2 = min(color_image.shape[1], x2)
        # y2 = min(color_image.shape[0], y2)

        # # Extract the color and depth images for the bounding box
        # cropped_color = color_image[y1:y2, x1:x2]
        # cropped_depth = depth_image[y1:y2, x1:x2].astype(np.uint8)

        # # Ensure the arrays are contiguous
        # cropped_color = np.ascontiguousarray(cropped_color)
        # cropped_depth = np.ascontiguousarray(cropped_depth)
        # Create a mask based on the cropped color image
        # import pdb
        # pdb.set_trace()
#       Show color_image, depth_image, color_mask
        # cv2.waitKey(0)
        mask = color_mask.any(axis=-1)
        # cv2.waitKey(0)
        #add 3rd dimension to mask
        arr_3d = np.expand_dims(mask, axis=-1)  # Add a new axis at the end
        arr_3d = np.tile(arr_3d, (1, 1, 3))        # Apply the mask to the depth image
        cropped_depth = np.where(mask, depth_image, np.nan).astype(np.uint8)
        cropped_color = np.where(arr_3d, color_image, 0).astype(np.uint8)
        # cv2.waitKey(0)
        # cropped_depth = masked_depth
        # Resize depth image to match the color image size if needed
        # if cropped_color.shape[:2] != cropped_depth.shape[:2]:
        #     cropped_depth = cv2.resize(cropped_depth, (cropped_color.shape[1], cropped_color.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Convert to Open3D format
        o3d_color = o3d.geometry.Image(cropped_color)
        o3d_depth = o3d.geometry.Image(cropped_depth)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale= 1000.0, convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.pinhole_camera_intrinsic, self.extrinsic)
        # o3d.visualization.draw_geometries([pcd])
        # denoised_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10,
        #                                                                     std_ratio=2.0)
        # import pdb
        # pdb.set_trace()
        # pcd = pcd.voxel_down_sample(voxel_size=0.001)
        pcd1, ind = pcd.remove_statistical_outlier(nb_neighbors=2.0, std_ratio=2.0)
        # inliers = pcd.select_by_index(ind)
        # o3d.visualization.draw_geometries([inliers])
        # outliers = pcd.select_by_index(ind, invert=True)
        
        # Compute the 3D bounding box
        bbox_3d = pcd1.get_axis_aligned_bounding_box()
        # bbox_3d = pcd1.get_oriented_bounding_box()
        # bbox_3d = inliers.get_axis_aligned_bounding_box()
        #give random color in format (R,G,B)
        random_color = self.rng.uniform(0, 1, size=(3,))
        bbox_3d.color = random_color
        # Visualize the bounding box (for debugging purposes)
        # o3d.visualization.draw_geometries([pcd, bbox_3d])
        # Optionally, return or save the bounding box for further processing
        return bbox_3d, pcd
