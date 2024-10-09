import os
import cv2
import numpy as np
from YOLOWorld import YOLOWorld
from DetectionDrawer import DetectionDrawer
from YOLOWorld import read_class_embeddings
from ultralytics import SAM



sam_model = SAM('/home/rameez/ew_rs/src/rl_detect/rl_detect/models/mobile_sam.pt')
model_path = '/home/rameez/ew_rs/src/rl_detect/rl_detect/models/yolov8s-worldv2_11.onnx'
embed_path = '/home/rameez/ew_rs/src/rl_detect/rl_detect/data/ew_embeddings_new_11.npz'

# Load class embeddings
class_embeddings, class_list = read_class_embeddings(embed_path)

# Initialize YOLO-World object detector
yoloworld_detector = YOLOWorld(model_path, conf_thres=0.1, iou_thres=0.5)
# Set up the RealSense D455 camera
drawer = DetectionDrawer(sam_model ,class_list)
   

#load image
color_image = cv2.imread('/home/rameez/ew_rs/src/rl_detect/rl_detect/Untitled.jpeg')
boxes, scores, class_ids = yoloworld_detector(color_image, class_embeddings)
semantic_mask = drawer(color_image, boxes, scores, class_ids)
#save image
cv2.imwrite('/home/rameez/ew_rs/src/rl_detect/rl_detect/output.jpg', semantic_mask)