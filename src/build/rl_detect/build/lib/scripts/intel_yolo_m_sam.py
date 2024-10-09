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

        # Initialize occupancy map image to same size as input image
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)  # Assuming input image size is 640x480

    def pixel_to_point(self, x, y, depth):
        intrin = self.cfg.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        pixel = [x, y]
        depth_point = rs.rs2_deproject_pixel_to_point(intrin, pixel, depth)
        return depth_point


    def run(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_image = depth_image * self.depth_scale #depth_image is in meters
            depth_image_uint8 = depth_image.astype(np.uint8)
            depth_image_uint8 = (depth_image * 255 / 10.0).astype(np.uint8) #divide by specif number eg 10
            depth_heatmap = cv2.applyColorMap(depth_image_uint8, cv2.COLORMAP_JET)
            boxes, scores, class_ids = self.yoloworld_detector(color_image, self.class_embeddings)
            combined_img, semantic_mask = self.drawer(color_image,depth_image, boxes, scores, class_ids)
            binary_mask = np.where(semantic_mask > 0, 255, 0).astype(np.uint8)
            print("Binary Mask Shape:", binary_mask.shape)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow("Color Image", color_image)
            cv2.imshow("Depth Image", depth_heatmap)
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
