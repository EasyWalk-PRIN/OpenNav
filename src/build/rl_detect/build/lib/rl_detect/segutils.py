from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
           [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
           [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
           [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
           [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
           [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
           [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
           [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
           [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
           [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
           [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
           [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
           [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
           [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
           [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
           [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
           [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
           [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
           [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
           [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
           [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
           [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
           [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
           [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
           [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
           [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
           [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
           [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
           [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
           [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
           [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
           [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
           [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
           [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
           [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
           [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
           [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
           [102, 255, 0], [92, 0, 255]]
CLASSES = ["wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass",
           "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair",
           "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
           "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
           "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper",
           "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table",
           "pillow", "screen door", "stairway", "river", "bridge", "bookcase", "blind", "coffee table",
           "toilet", "flower", "book", "hill", "bench", "countertop", "stove", "palm", "kitchen island",
           "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel", "light",
           "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television", "airplane",
           "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle",
           "buffet", "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer",
           "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike",
           "cradle", "oven", "ball", "food", "step", "tank", "trade name", "microwave", "pot", "animal",
           "bicycle", "lake", "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase",
           "traffic light", "tray", "ashcan", "fan", "pier", "crane", "pier", "headboard", "outlet",
           "salt shaker", "paper towel dispenser", "soap dispenser", "stretcher", "cart", "fire extinguisher",
           "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food", "step",
           "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen",
           "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan",
           "pier", "crane", "pier", "headboard", "outlet", "salt shaker", "paper towel dispenser",
           "soap dispenser", "stretcher", "cart", "fire extinguisher"]
def preprocess(
    img: np.ndarray, input_size: Tuple[int, int] = (512, 512)
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess image for inference."""
    img_shape = img.shape[:2]
    # Resize
    resized_img = cv2.resize(img, input_size)

    # Normalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    resized_img = (resized_img - mean) / std

    return resized_img, img_shape

def build_session(onnx_file: str, device: str = 'cpu') -> ort.InferenceSession:
    """Build onnxruntime session.

    Args:
        onnx_file (str): ONNX file path.
        device (str): Device type for inference.

    Returns:
        sess (ort.InferenceSession): ONNXRuntime session.
    """
    providers = ['CPUExecutionProvider'
                 ] if device == 'cpu' else ['CUDAExecutionProvider']
    sess = ort.InferenceSession(path_or_bytes=onnx_file, providers=providers)

    return sess

def inference(sess: ort.InferenceSession, img: np.ndarray) -> np.ndarray:
    """Inference RTMPose model.

    Args:
        sess (ort.InferenceSession): ONNXRuntime session.
        img (np.ndarray): Input image in shape.

    Returns:
        outputs (np.ndarray): Output of RTMPose model.
    """
    # build input
    input_img = [img.transpose(2, 0, 1).astype(np.float32)]

    # build output
    sess_input = {sess.get_inputs()[0].name: input_img}
    sess_output = []
    for out in sess.get_outputs():
        sess_output.append(out.name)

    # inference
    outputs = sess.run(output_names=sess_output, input_feed=sess_input)

    return outputs

def postprocess(outputs: List[np.ndarray],
                origin_shape: Tuple[int, int]) -> np.ndarray:
    """Postprocess outputs of PP_Mobileseg model.

    Args:
        outputs (List[np.ndarray]): Outputs of PP_Mobileseg model.
        origin_shape (Tuple[int, int]): Input size of PP_Mobileseg model.

    Returns:
        seg_map (np.ndarray): Segmentation map.
    """
    seg_map = outputs[0][0][0]
    seg_map = cv2.resize(seg_map.astype(np.float32), origin_shape)
    return seg_map

def visualize(seg_map: np.ndarray) -> np.ndarray:
    palette = np.array(PALETTE)
    color_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3),
                         dtype=np.uint8)

    for label, color in enumerate(palette):
        color_seg[seg_map == label, :] = color
        # Find the center of the class region for text placement
        mask = (seg_map == label)
        if mask.any():
            y, x = np.where(mask)
            center_y, center_x = int(np.mean(y)), int(np.mean(x))
            cv2.putText(color_seg, CLASSES[label], (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # convert to BGR
    color_seg = color_seg[..., ::-1]

    return color_seg