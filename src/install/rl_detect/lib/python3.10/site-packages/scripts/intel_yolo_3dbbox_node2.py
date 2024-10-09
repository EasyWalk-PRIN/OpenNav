import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import pyrealsense2 as rs
from rl_detect.YOLOWorld import YOLOWorld
from rl_detect.DetectionDrawer import DetectionDrawer
from rl_detect.YOLOWorld import read_class_embeddings
from ultralytics import SAM
import open3d as o3d
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
import pdb
import time
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
        self.sam_model.to('cuda')
        self.model_path = self.get_parameter('model_path').value
        self.embed_path = self.get_parameter('embed_path').value

        # Load class embeddings
        self.class_embeddings, self.class_list = read_class_embeddings(self.embed_path)

        # Initialize YOLO-World object detector
        self.yoloworld_detector = YOLOWorld(self.model_path, conf_thres=0.1, iou_thres=0.5)

        self.bridge = CvBridge()

        # ROS 2 subscriptions for depth and color images
        self.color_subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.color_callback,
            10)
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/depth/image_rect_raw',
            self.depth_callback,
            10)
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_callback,
            10)

        self.publisher = self.create_publisher(Image, 'yolo_detected_objects', 10)
        self.marker_array_publisher = self.create_publisher(MarkerArray, 'yolo_detected_objects_marker_array', 10)
        self.color_image = None
        self.depth_image = None
        self.camera_info = None

        self.drawer = None
        self.bboxes = []

    def color_callback(self, msg):
        self.get_logger().info('Receiving color image')
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.color_frame_id = msg.header.frame_id
        self.process_images()

    def depth_callback(self, msg):
        self.get_logger().info('Receiving depth image')
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth_frame_id = msg.header.frame_id
        self.process_images()

    def camera_info_callback(self, msg):
        self.get_logger().info('Receiving camera info')
        self.camera_info = msg
        intrinsics = rs.intrinsics()
        intrinsics.ppx = msg.k[2]
        intrinsics.ppy = msg.k[5]
        intrinsics.fx = msg.k[0]
        intrinsics.fy = msg.k[4]
        intrinsics.model = rs.distortion.none
        intrinsics.coeffs = [i for i in msg.d]
        self.intr = intrinsics

        if self.drawer is None:
            self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                self.intr.width, self.intr.height, self.intr.fx, self.intr.fy, self.intr.ppx, self.intr.ppy)
            self.extrinsic = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            self.drawer = DetectionDrawer(self.sam_model, self.class_list, self.pinhole_camera_intrinsic, self.extrinsic, self.intr)

    def process_images(self):
        if self.color_image is not None and self.depth_image is not None and self.camera_info is not None:
            self.get_logger().info('Processing images with camera parameters')
            depth_image = self.depth_image
            color_image = self.color_image

            try:
                # start_time = time.time()
                boxes, scores, class_ids = self.yoloworld_detector(color_image, self.class_embeddings)
                bbox3d = self.drawer(color_image, depth_image, boxes, scores, class_ids)
                header = Header()
                header.stamp = self.get_clock().now().to_msg()   
                header.frame_id = self.color_frame_id  # same frame_id as the colorframe         
                
                # Publish markers for each bounding box
                marker_array = MarkerArray()
                for i, bbox in enumerate(bbox3d):
                    marker = Marker()
                    marker.header = header 
                    marker.ns = "yolo_detected_objects"
                    marker.id = i
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = bbox.get_center()
                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = 0.0
                    marker.pose.orientation.z = 0.0
                    marker.pose.orientation.w = 1.0
                    marker.scale.x, marker.scale.y, marker.scale.z = bbox.get_extent()
                    marker.color.a = 0.2  # Transparency
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0

                    marker_array.markers.append(marker)
                self.marker_array_publisher.publish(marker_array)
                # end_time = time.time()
                # #time in milliseconds
                # elapsed_time = (end_time - start_time) * 1000
                # print(f"Time taken to process the image and get the bounding boxes: {elapsed_time} ms")

            except Exception as e:
                self.get_logger().error(f"Error processing images: {e}")
                pass

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
