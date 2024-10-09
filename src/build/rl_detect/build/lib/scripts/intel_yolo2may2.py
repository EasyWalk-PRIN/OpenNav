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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', '/home/rameez/ew_rs/src/rl_detect/rl_detect/models/yolov8s-worldv2.onnx'),
                ('embed_path', '/home/rameez/ew_rs/src/rl_detect/rl_detect/data/ew_embeddings_new.npz')
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
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.cfg = self.pipeline.start(self.config)
        self.drawer = DetectionDrawer(self.sam_model ,self.class_list)
        self.bridge = CvBridge()

        self.publisher = self.create_publisher(Image, 'yolo_detected_objects', 10)
        self.depth_scale = self.cfg.get_device().first_depth_sensor().get_depth_scale()

        # Initialize occupancy map using point cloud
        self.pointcloud = rs.pointcloud()
        self.depth_intrin = None
        self.occupancy_map = np.zeros((480, 640), dtype=np.uint8)  # Assuming input image size is 640x480



    def run(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_image = depth_image * self.depth_scale
            depth_image_uint8 = (depth_image * 255 / np.max(depth_image)).astype(np.uint8)
            depth_heatmap = cv2.applyColorMap(depth_image_uint8, cv2.COLORMAP_JET)
            boxes, scores, class_ids = self.yoloworld_detector(color_image, self.class_embeddings)
            combined_img, semantic_mask = self.drawer(color_image, depth_image, boxes, scores, class_ids)
            binary_mask = np.where(semantic_mask > 0, 255, 0).astype(np.uint8)
            print("Binary Mask Shape:", binary_mask.shape)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(depth_colormap, 0.5, cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR), 0.5, 0)

            # Update occupancy map with detected objects
            self.occupancy_map.fill(0)  # Reset occupancy map
            points = self.pointcloud.calculate(depth_frame)
            vertices = np.asarray(points.get_vertices())
            for box, score, class_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(self.occupancy_map, (x1, y1), (x2, y2), (255, 255, 255), -1)  # Highlight detected object on map

                # Calculate depth
                depth_roi = depth_image[y1:y2, x1:x2]
                mean_depth = np.mean(depth_roi)

                # Convert pixel coordinates to real-world coordinates
                depth_point = self.pixel_to_point((x1 + x2) // 2, (y1 + y2) // 2, mean_depth)

                # Update occupancy map with object location
                self.update_occupancy_map(depth_point)

                # Print real-world coordinates of the detected object
                self.get_logger().info(f"Object: {self.class_list[int(class_id)]}, Depth: {mean_depth:.2f} m, Coordinates: {depth_point}")

                # Annotate object name and depth
                label = f"{self.class_list[int(class_id)]}: {mean_depth:.2f} m"
                cv2.putText(self.occupancy_map, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Convert the occupancy map image to ROS Image message and publish
            out_msg = self.bridge.cv2_to_imgmsg(self.occupancy_map, 'mono8')
            self.publisher.publish(out_msg)

            # Plot the occupancy map in 3D
            ax.clear()
            X, Y = np.meshgrid(range(self.occupancy_map.shape[1]), range(self.occupancy_map.shape[0]))
            ax.plot_surface(X, Y, self.occupancy_map, cmap='binary')
            plt.pause(0.001)

            cv2.imshow('Overlay', overlay)
            cv2.imshow("Color Image", color_image)
            cv2.imshow("Depth Image", depth_heatmap)
            cv2.imshow("Combined Image", combined_img)
            cv2.imshow("Occupancy Map", self.occupancy_map)
            cv2.waitKey(1)


    def pixel_to_point(self, x, y, depth):
        if self.depth_intrin is None:
            self.depth_intrin = self.cfg.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        pixel = [x, y]
        depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, pixel, depth)
        return depth_point

    def update_occupancy_map(self, point):
        # Update the occupancy map using the point cloud
        x, y, z = point
        x_idx = int(x * 1000)  # convert meters to millimeters
        y_idx = int(y * 1000)
        
        # Check if indices are within the bounds of the occupancy map
        if 0 <= y_idx < self.occupancy_map.shape[0] and 0 <= x_idx < self.occupancy_map.shape[1]:
            self.occupancy_map[y_idx, x_idx] = 255  # Mark the point on the occupancy map
        else:
            self.get_logger().warning("Point coordinates out of bounds for occupancy map.")


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
