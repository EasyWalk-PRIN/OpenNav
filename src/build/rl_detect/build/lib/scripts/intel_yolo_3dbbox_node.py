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
import pdb
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tf_transformations
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header

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
        # self.marker_publisher = self.create_publisher(Marker, 'yolo_detected_objects_marker', 10)
        self.marker_array_publisher = self.create_publisher(MarkerArray, 'yolo_detected_objects_marker_array', 10)
        self.point_cloud_publisher = self.create_publisher(PointCloud2, '/camera/depth/points', 10)

        self.color_image = None
        self.depth_image = None
        self.camera_info = None

        self.drawer = None

        # Open3D visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='YOLO Detection Visualization')
        self.point_cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.point_cloud)
        self.bboxes = []

    def color_callback(self, msg):
        self.get_logger().info('Receiving color image')
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.process_images()

    def depth_callback(self, msg):
        self.get_logger().info('Receiving depth image')
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.process_images()

    def camera_info_callback(self, msg):
        self.get_logger().info('Receiving camera info')
        self.camera_info = msg
        intrinsics = rs.intrinsics()
        intrinsics.width = 848
        intrinsics.height = 480
        scale_x = 848 / msg.width
        scale_y = 480 / msg.height
        intrinsics.ppx = msg.k[2] * scale_x
        intrinsics.ppy = msg.k[5] * scale_y
        intrinsics.fx = msg.k[0] * scale_x
        intrinsics.fy = msg.k[4] * scale_y
        intrinsics.model = rs.distortion.none  # Change if necessary
        intrinsics.coeffs = [i for i in msg.d]
        self.intr = intrinsics
        # pdb.set_trace()
        if self.drawer is None:
            self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                self.intr.width, self.intr.height, self.intr.fx, self.intr.fy, self.intr.ppx, self.intr.ppy)
            self.extrinsic = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            self.drawer = DetectionDrawer(self.sam_model, self.class_list, self.pinhole_camera_intrinsic, self.extrinsic, self.intr)

    def process_images(self):
        if self.color_image is not None and self.depth_image is not None and self.camera_info is not None:
            self.get_logger().info('Processing images with camera parameters')
            depth_image = self.depth_image
            h, w = depth_image.shape
            color_image = cv2.resize(self.color_image, (w, h), interpolation=cv2.INTER_NEAREST)
            boxes, scores, class_ids = self.yoloworld_detector(color_image, self.class_embeddings)
            combined_img, semantic_mask, pcd, bbox3d = self.drawer(color_image, depth_image, boxes, scores, class_ids)
            
            # Publish point cloud data
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "camera_color_optical_frame"  # Adjust frame_id if necessary
            
            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            
            cloud_msg = PointCloud2()
            cloud_msg.header = header
            cloud_msg.fields = fields
            cloud_msg.height = 1
            cloud_msg.width = len(pcd.points)
            cloud_msg.is_bigendian = False
            cloud_msg.point_step = 12
            cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
            cloud_msg.is_dense = True if np.isfinite(pcd.points).all() else False
            cloud_msg.data = np.asarray(pcd.points, np.float32).tobytes()
            
            self.point_cloud_publisher.publish(cloud_msg)
            
            # Publish markers for each bounding box
            marker_array = MarkerArray()
            for i, bbox in enumerate(bbox3d):
                marker = Marker()
                marker.header = header  # Set the header for the marker
                marker.ns = "yolo_detected_objects"
                marker.id = i
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = bbox.get_center()
                obb_rotation_matrix = bbox.R
                rotation_quaternion = tf_transformations.quaternion_from_matrix([
                    [obb_rotation_matrix[0, 0], obb_rotation_matrix[0, 1], obb_rotation_matrix[0, 2], 0],
                    [obb_rotation_matrix[1, 0], obb_rotation_matrix[1, 1], obb_rotation_matrix[1, 2], 0],
                    [obb_rotation_matrix[2, 0], obb_rotation_matrix[2, 1], obb_rotation_matrix[2, 2], 0],
                    [0, 0, 0, 1]
                ])
                marker.pose.orientation.x = rotation_quaternion[0]
                marker.pose.orientation.y = rotation_quaternion[1]
                marker.pose.orientation.z = rotation_quaternion[2]
                marker.pose.orientation.w = rotation_quaternion[3]
                marker.scale.x, marker.scale.y, marker.scale.z = bbox.extent
                marker.color.a = 0.5  # Transparency
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0

                marker_array.markers.append(marker)
            self.marker_array_publisher.publish(marker_array)
            
            # Update Open3D visualization and show combined image
            self.point_cloud.points = pcd.points
            self.point_cloud.colors = pcd.colors
            self.vis.update_geometry(self.point_cloud)

            for bbox in self.bboxes:
                self.vis.remove_geometry(bbox)
            self.bboxes = []

            for bbox in bbox3d:
                bbox_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
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
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
