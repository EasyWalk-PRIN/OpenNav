import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import matplotlib.pyplot as plt

from rl_detect.YOLOWorld import YOLOWorld
from rl_detect.DetectionDrawer import DetectionDrawer
from rl_detect.YOLOWorld import read_class_embeddings

class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', '/home/rameez/ew_rs/src/rl_detect/rl_detect/models/yolov8x-worldv2.onnx'),
                ('embed_path', '/home/rameez/ew_rs/src/rl_detect/rl_detect/data/ew_embeddings.npz')
            ]
        )

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
        self.pipeline.start(self.config)
        self.drawer = DetectionDrawer(self.class_list)
        self.bridge = CvBridge()

        self.publisher = self.create_publisher(Image, 'yolo_detected_objects', 10)
        self.depth_scale = 0.0010000000474974513

    def run(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_image = depth_image * self.depth_scale
            boxes, scores, class_ids = self.yoloworld_detector(color_image, self.class_embeddings)
            depth_image_uint8 = (depth_image * 255 / np.max(depth_image)).astype(np.uint8)
            depth_heatmap = cv2.applyColorMap(depth_image_uint8, cv2.COLORMAP_JET)
            # Integrate 3D position estimation
            positions_3d = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers
                depth_roi = depth_image[y1:y2, x1:x2]
                if depth_roi.size == 0:
                    # Skip processing if the depth_roi is empty
                    continue
                average_depth = depth_roi.mean()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                z_3d = average_depth
                x_3d = (cx - color_image.shape[1] / 2) * z_3d / (color_image.shape[1] / 2)
                y_3d = (cy - color_image.shape[0] / 2) * z_3d / (color_image.shape[0] / 2)
                positions_3d.append((x_3d, y_3d, z_3d))

            # Overlay 3D positions on depth image
            for position in positions_3d:
                x, y, z = position
                if np.isnan(z) or np.isnan(x) or np.isnan(y):
                    # Skip processing if depth value is NaN
                    continue
                # Convert 3D position to pixel coordinates on the depth image
                px = int((x * (640 / 2)) / z + 320)
                py = int((y * (480 / 2)) / z + 240)
                # Draw a marker at the pixel coordinates on the depth heatmap and put text describing depth value
                cv2.putText(depth_heatmap, f'{z:.2f}m', (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.circle(depth_heatmap, (px, py), 5, (0, 0, 255), -1)

            
            
            # Convert depth image to uint8 format
            depth_image_uint8 = (depth_image * 255 / np.max(depth_image)).astype(np.uint8)

            # Publish the depth image with overlaid markers
            out_msg = self.bridge.cv2_to_imgmsg(depth_heatmap, 'bgr8')
            self.publisher.publish(out_msg)
            combined_img = self.drawer(color_image,depth_image, boxes, scores, class_ids)
            out_msg = self.bridge.cv2_to_imgmsg(combined_img, 'bgr8')
            self.publisher.publish(out_msg)

            cv2.imshow("Color Image", color_image)
            cv2.imshow("Depth Image", depth_heatmap)
            cv2.imshow("Combined Image", combined_img)
            cv2.waitKey(1)
            # # Display the depth image with overlaid markers
            # cv2.imshow("Depth Image with Markers", depth_image)
            # cv2.waitKey(1)

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
