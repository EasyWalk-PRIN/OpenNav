import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import pyrealsense2 as rs
from rl_detect.DetectionDrawer import DetectionDrawer
from ultralytics import SAM
from ultralytics import YOLO
import open3d as o3d
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('segmentation', ''),
                ('detection', ''),
                ('classes', ''),
                ('output_markers', ''),  
                ('output_pointcloud', ''),  
                ('output_detection_3d', '') 
            ]
        )
        self.sam_model = SAM(self.get_parameter('segmentation').value)
        self.sam_model.to('cuda')

        classes_param = self.get_parameter('classes').get_parameter_value().string_value
        self.classes = classes_param.split(',')
        self.yoloworld_detector = YOLO(self.get_parameter('detection').value)
        self.yoloworld_detector.to('cuda')
        self.yoloworld_detector.set_classes(self.classes)
        print("Detecting classes:", self.classes)

        self.bridge = CvBridge()

        # ROS 2 subscriptions
        self.color_subscription = self.create_subscription(
            Image, '/camera/color/image_raw', self.color_callback, 10)
        self.depth_subscription = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.camera_info_subscription = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)

        # Publishers
        self.marker_array_publisher = self.create_publisher(MarkerArray, 'yolo_detected_objects_marker_array', 10)
        self.point_cloud_publisher = self.create_publisher(PointCloud2, '/camera/depth/segment', 10)
        self.detection_3d_publisher = self.create_publisher(Detection3DArray, '/camera/detection_3d', 10)

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

    def process_images(self):
        if self.color_image is not None and self.depth_image is not None and self.camera_info is not None:
            self.get_logger().info('Processing images with camera parameters')
            depth_image = self.depth_image
            color_image = self.color_image
            h, w = depth_image.shape
            color_image = cv2.resize(self.color_image, (w, h), interpolation=cv2.INTER_NEAREST)
            depth_image_normalized = cv2.normalize(depth_image / 1000.0, None, 0, 255, cv2.NORM_MINMAX)
            depth_image_normalized = depth_image_normalized.astype('uint8')
            color_mapped_image = cv2.applyColorMap(depth_image_normalized, cv2.COLORMAP_JET)

            try:
                detection_results = self.yoloworld_detector(color_image, conf=0.3, iou=0.5)
                boxes = detection_results[0].boxes.xyxy.tolist()
                scores = detection_results[0].boxes.conf.cpu().numpy()
                class_ids = detection_results[0].boxes.cls.cpu().numpy().astype(int)

                bbox3d, pcd = self.drawer(color_image, depth_image, boxes, scores, class_ids)

                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = self.color_frame_id

                # Check user preferences and publish accordingly
                if self.get_parameter('output_pointcloud').value == "True":
                    self.publish_pointcloud(header, pcd)
                if self.get_parameter('output_markers').value == "True":
                    self.publish_markers(header, bbox3d, class_ids)
                if self.get_parameter('output_detection_3d').value == "True":
                    self.publish_3d_detections(header, bbox3d, scores, class_ids)

            except Exception as e:
                self.get_logger().error(f"Error processing images: {e}")
                pass

    def publish_pointcloud(self, header, pcd):
        # Prepare and publish point cloud message
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.fields = fields
        cloud_msg.height = 1
        cloud_msg.width = len(pcd.points)
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 16  
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

    def publish_markers(self, header, bbox3d, class_ids):
        # Publish marker array with different colors for each class
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
            
            class_id = class_ids[i]
            color = self.get_color_for_class(class_id)
            marker.color.a = 0.6  # Transparency
            marker.color.r, marker.color.g, marker.color.b = color

            marker_array.markers.append(marker)
        self.marker_array_publisher.publish(marker_array)

    def publish_3d_detections(self, header, bbox3d, scores, class_ids):
        detection_array_msg = Detection3DArray()
        detection_array_msg.header = header

        for i, bbox in enumerate(bbox3d):
            detection_msg = Detection3D()

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = self.classes[class_ids[i]] 
            hypothesis.hypothesis.score = float(scores[i])  

            center_x, center_y, center_z = bbox.get_center()
            extent_x, extent_y, extent_z = bbox.get_extent()

            hypothesis.pose.pose.position.x = center_x
            hypothesis.pose.pose.position.y = center_y
            hypothesis.pose.pose.position.z = center_z
            hypothesis.pose.pose.orientation.x = 0.0
            hypothesis.pose.pose.orientation.y = 0.0
            hypothesis.pose.pose.orientation.z = 0.0
            hypothesis.pose.pose.orientation.w = 1.0
            detection_msg.results.append(hypothesis)

            detection_msg.bbox.size.x = extent_x
            detection_msg.bbox.size.y = extent_y
            detection_msg.bbox.size.z = extent_z

            detection_array_msg.detections.append(detection_msg)

        # Publish the 3D detections
        self.detection_3d_publisher.publish(detection_array_msg)


    def get_color_for_class(self, class_id):
        # Generate color for each class (simple mapping)
        color_map = [
            (1.0, 0.0, 0.0),  # Red for class 0
            (0.0, 1.0, 0.0),  # Green for class 1
            (0.0, 0.0, 1.0),  # Blue for class 2
            (1.0, 1.0, 0.0),  # Yellow for class 3
            (1.0, 0.0, 1.0),  # Magenta for class 4
            (0.0, 1.0, 1.0)   # Cyan for class 5 # Add more colors if needed
        ]
        return color_map[class_id % len(color_map)]


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
