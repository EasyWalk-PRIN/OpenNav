import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
#print the current working directory
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
        # self.get_logger().info("Detecting classes: %s", self.class_list)

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

            combined_img = self.drawer(color_image,depth_image, boxes, scores, class_ids)
            out_msg = self.bridge.cv2_to_imgmsg(combined_img, 'bgr8')
            self.publisher.publish(out_msg)

            cv2.imshow("Color Image", color_image)
            cv2.imshow("Depth Image", depth_image)
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
