import cv2
import numpy as np


class DetectionDrawer():

    def __init__(self, sam,class_names):
        self.sam_model = sam
        self.class_names = class_names
        self.rng = np.random.default_rng(3)
        self.colors = self.get_colors(len(class_names))

    def __call__(self, image, depth_image,boxes, scores, class_ids, mask_alpha=0.3):
        return self.draw_detections(image, depth_image, boxes, scores, class_ids, mask_alpha)

    def update_class_names(self, class_names):
        for i, class_name in enumerate(class_names):
            if class_name not in self.class_names:
                self.colors[i] = self.get_colors(1)
            else:
                self.colors[i] = self.colors[list(self.class_names).index(class_name)]

        self.class_names = class_names
    def bool_mask_to_int(self, mask, true_value=(255, 0, 0)):
        """
        Convert boolean mask to integer representing colors.

        Parameters:
            mask (numpy.ndarray or list): Boolean mask.
            true_value (tuple): RGB color tuple for True values in the mask.

        Returns:
            numpy.ndarray or list: Integer mask representing colors.
        """
        true_value = np.array(true_value)
        return mask.astype(np.uint8)[:, :, None] * true_value

    def draw_detections(self, image, depth_image, boxes, scores, class_ids, mask_alpha=0.3):
        if class_ids.shape[0] == 0:
            return image
        mask_img = np.zeros_like(image)  # Initialize mask image
        det_img = image.copy()
        
        img_height, img_width = image.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        classes = self.class_names[class_ids]
        colors = self.colors[class_ids]
        results = self.sam_model.predict(image, bboxes=boxes)[0]
        i = 0
        for box, score, label, color in zip(boxes, scores, classes, colors):
            x1, y1, x2, y2 = box.astype(int)
            object_depth = np.median(depth_image[int(y1):int(y2), int(x1):int(x2)])

            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
            mask = results[i].masks.data.cpu().numpy()  # 1, 480, 640
            i += 1
            mask = mask.squeeze(axis=0)  # Remove the first dimension
            color_mask = self.bool_mask_to_int(mask, true_value=color)
            color_mask = color_mask.astype(mask_img.dtype)
            mask_img = cv2.add(mask_img, color_mask)
            caption = f'{label} {score*100:.2f}%' + f' {object_depth:.2f}m'
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)

            cv2.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
            cv2.putText(det_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

            # cv2.putText(mask_img, caption, (x1, y1),
            #             cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0), mask_img



    def get_colors(self, num_classes):
        return self.rng.uniform(0, 255, size=(num_classes, 3))


