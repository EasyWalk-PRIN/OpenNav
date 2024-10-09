import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
from rl_detect.YOLOWorld import YOLOWorld
from rl_detect.DetectionDrawer import DetectionDrawer
from rl_detect.YOLOWorld import read_class_embeddings
from ultralytics import SAM
import math
import time
import pdb
import open3d as o3d

import open3d as o3d
import numpy as np

class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', '/home/rameez/ew_rs/src/rl_detect/rl_detect/models/yolov8s-worldv2_11.onnx'),
                ('embed_path', '/home/rameez/ew_rs/src/rl_detect/rl_detect/data/ew_embeddings_new_11.npz')
            ]
        )
        self.sam_model = SAM('/home/rameez/ew_rs/src/rl_detect/rl_detect/models/mobile_sam.pt')
        self.model_path = self.get_parameter('model_path').value
        self.embed_path = self.get_parameter('embed_path').value

        # Load class embeddings
        self.class_embeddings, self.class_list = read_class_embeddings(self.embed_path)

        # Initialize YOLO-World object detector
        self.yoloworld_detector = YOLOWorld(self.model_path, conf_thres=0.1, iou_thres=0.5)
        # Set up the RealSense D455 camera
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.cfg = self.pipeline.start(self.config)
        self.bridge = CvBridge()

        self.publisher = self.create_publisher(Image, 'yolo_detected_objects', 10)
        self.depth_scale = self.cfg.get_device().first_depth_sensor().get_depth_scale()

        # Initialize occupancy map using point cloud
        self.depth_intrin = None
        # Get stream profile and camera intrinsics
        profile = self.pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        # Processing blocks
        self.pc = rs.pointcloud()
        self.decimate = rs.decimation_filter()
        self.colorizer = rs.colorizer()

        self.intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.intr.width, self.intr.height, self.intr.fx, self.intr.fy, self.intr.ppx, self.intr.ppy)
        self.extrinsic = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        self.out = np.empty((h, w, 3), dtype=np.uint8)
        self.drawer = DetectionDrawer(self.sam_model, self.class_list, self.pinhole_camera_intrinsic, self.extrinsic, self.intr)
        
        # Open3D visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='YOLO Detection Visualization')
        self.point_cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.point_cloud)
        self.bboxes = []

    def run(self):
        while rclpy.ok():
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
            w, h = depth_intrinsics.width, depth_intrinsics.height

            depth_image = np.asanyarray(depth_frame.get_data())
            o3d_depth = o3d.geometry.Image(depth_image)
            color_image = np.asanyarray(color_frame.get_data())
            
            depth_image = depth_image / 10.0
            color_image = cv2.resize(color_image, (w, h), interpolation=cv2.INTER_NEAREST)
            boxes, scores, class_ids = self.yoloworld_detector(color_image, self.class_embeddings)
            combined_img, semantic_mask, pcd, bbox3d = self.drawer(color_image, depth_image, boxes, scores, class_ids)

            # Convert pcd (point cloud) to Open3D format and update the visualizer
            self.point_cloud.points = pcd.points
            self.point_cloud.colors = pcd.colors
            self.vis.update_geometry(self.point_cloud)

            # Remove old bounding boxes
            for bbox in self.bboxes:
                self.vis.remove_geometry(bbox)
            self.bboxes = []

            # Add new bounding boxes
            for bbox in bbox3d:
                bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
                # bbox_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
                self.bboxes.append(bbox_lines)
                self.vis.add_geometry(bbox_lines)

            self.vis.poll_events()
            self.vis.update_renderer()

            cv2.imshow("Combined Image", combined_img)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectorNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
