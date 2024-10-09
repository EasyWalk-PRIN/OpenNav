import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
#change the current working directory to the directory where the file is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from yoloworld import YOLOWorld, DetectionDrawer, read_class_embeddings
import numpy as np
import ros_numpy

model_path = "models/yolov8x-worldv2.onnx"
embed_path = "data/ew_embeddings.npz"

# Load class embeddings
class_embeddings, class_list = read_class_embeddings(embed_path)
print("Detecting classes:", class_list)
drawer = DetectionDrawer(class_list)
# Initialize YOLO-World object detector
yoloworld_detector = YOLOWorld(model_path, conf_thres=0.1, iou_thres=0.5)

class IntelSubscriber(Node):
    def __init__(self):
        super().__init__("intel_subscriber")
        self.subscription_rgb = self.create_subscription(Image, "rgb_frame", self.rgb_frame_callback, 10)
        self.br_rgb = CvBridge()


    def rgb_frame_callback(self, data):
        self.get_logger().warning("Receiving RGB frame")
        # current_frame = self.br_rgb.imgmsg_to_cv2(data)
        current_frame = ros_numpy.numpify(data)
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        boxes, scores, class_ids = yoloworld_detector(current_frame, class_embeddings)
        combined_img = drawer(current_frame , boxes, scores, class_ids)
        #show combined image and current frame side by side
        cv2.imshow("Combined", combined_img)
        cv2.imshow("RGB", current_frame)
        cv2.waitKey(1)




def main(args = None):
    rclpy.init(args = args)
    intel_subscriber = IntelSubscriber()
    rclpy.spin(intel_subscriber)
    intel_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()