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
from ultralytics import YOLO
import open3d as o3d
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
import pdb
import time
class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('segmentation', ''),
                ('detection', ''),
                ('classes', '')
            ]
        )        
        self.sam_model = SAM(self.get_parameter('segmentation').value)
        self.sam_model.to('cuda')

        # Initialize YOLO-World object detector
        classes_param = self.get_parameter('classes').get_parameter_value().string_value
        # Split the comma-separated string into a list
        self.classes = classes_param.split(',')
        self.yoloworld_detector = YOLO(self.get_parameter('detection').value)
        self.yoloworld_detector.set_classes(self.classes)
        print("Detecting classes:", self.classes)

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
        self.point_cloud_publisher = self.create_publisher(PointCloud2, '/camera/depth/segment', 10)
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
            self.drawer = DetectionDrawer(self.sam_model, self.classes, self.pinhole_camera_intrinsic, self.extrinsic, self.intr)
    def save_point_cloud_as_image(self, pcd, image_path, width=640, height=480):
        # Convert Open3D point cloud to numpy arrays
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        # Create an empty image
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Define camera intrinsic parameters
        fx = fy = width  # Focal length
        cx = width / 2  # Principal point x
        cy = height / 2  # Principal point y

        # Project 3D points to 2D
        for point, color in zip(points, colors):
            # Convert point to camera coordinates
            u = int(fx * point[0] / point[2] + cx)
            v = int(fy * point[1] / point[2] + cy)
            if 0 <= u < width and 0 <= v < height:
                image[v, u] = (color * 255).astype(np.uint8)

        # Save the image using OpenCV
        cv2.imwrite(image_path, image)
    def process_images(self):
        if self.color_image is not None and self.depth_image is not None and self.camera_info is not None:
            self.get_logger().info('Processing images with camera parameters')
            depth_image = self.depth_image
            color_image = self.color_image
            h, w = depth_image.shape
            color_image = cv2.resize(self.color_image, (w, h), interpolation=cv2.INTER_NEAREST)
            depth_image_normalized = cv2.normalize(depth_image/1000.0, None, 0, 255, cv2.NORM_MINMAX)
            # Convert the normalized image to uint8 (8-bit) type
            depth_image_normalized = depth_image_normalized.astype('uint8')
            
            # Apply a colormap
            color_mapped_image = cv2.applyColorMap(depth_image_normalized, cv2.COLORMAP_JET)
    
            try:
                # start_time = time.time()
                detection_results = self.yoloworld_detector(color_image, conf=0.3, iou=0.5)
                boxes = detection_results[0].boxes.xyxy.tolist()
                scores = detection_results[0].boxes.conf.cpu().numpy()
                class_ids = detection_results[0].boxes.cls.cpu().numpy().astype(int)
                bbox3d, pcd = self.drawer(color_image, depth_image, boxes, scores, class_ids)
                header = Header()
                header.stamp = self.get_clock().now().to_msg()   
                header.frame_id = self.color_frame_id  # same frame_id as the colorframe         
                
                #publish point cloud
                # Define fields for PointCloud2 message
                fields = [
                    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                    PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
                ]

                # Create the PointCloud2 message
                cloud_msg = PointCloud2()
                cloud_msg.header = header
                cloud_msg.fields = fields
                cloud_msg.height = 1
                cloud_msg.width = len(pcd.points)
                cloud_msg.is_bigendian = False
                cloud_msg.point_step = 16  # 12 bytes for x, y, z + 4 bytes for rgb
                cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
                cloud_msg.is_dense = True if np.isfinite(pcd.points).all() else False

                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                rgb_colors = np.floor(colors * 255).astype(np.uint8)
                rgb_int = (rgb_colors[:, 0].astype(np.uint32) << 16) | (rgb_colors[:, 1].astype(np.uint32) << 8) | (rgb_colors[:, 2].astype(np.uint32))

                points_with_rgb = np.zeros((points.shape[0], 4), dtype=np.float32)
                points_with_rgb[:, 0:3] = points
                points_with_rgb[:, 3] = rgb_int.view(np.float32)
                cloud_data = points_with_rgb.tobytes()
                cloud_msg.data = cloud_data
                self.point_cloud_publisher.publish(cloud_msg)                
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
