import pyrealsense2 as rs
import numpy as np
import cv2

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 30)

# Start streaming
pipeline.start(config)

# Define downsampling factor
DOWNSAMPLE_FACTOR = 4

try:
    while True:
        # Wait for a coherent pair of frames: depth and infrared
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        ir_frame = frames.get_infrared_frame()

        if not depth_frame or not ir_frame:
            continue

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        ir_image = np.asanyarray(ir_frame.get_data())

        # Downsample depth image
        new_width = depth_frame.get_width() // DOWNSAMPLE_FACTOR
        new_height = depth_frame.get_height() // DOWNSAMPLE_FACTOR
        downsampled_depth = np.zeros((new_height, new_width), dtype=np.uint16)

        for y in range(0, new_height):
            for x in range(0, new_width):
                min_value = np.max(depth_image)

                for i in range(0, DOWNSAMPLE_FACTOR):
                    for j in range(0, DOWNSAMPLE_FACTOR):
                        pixel = depth_image[y * DOWNSAMPLE_FACTOR + i, x * DOWNSAMPLE_FACTOR + j]
                        if pixel != 0:
                            min_value = min(min_value, pixel)

                downsampled_depth[y, x] = min_value if min_value != np.max(depth_image) else 0

        # Apply edge filter
        scharr_x = cv2.Scharr(ir_image, cv2.CV_16S, 1, 0)
        abs_scharr_x = cv2.convertScaleAbs(scharr_x)
        scharr_y = cv2.Scharr(ir_image, cv2.CV_16S, 0, 1)
        abs_scharr_y = cv2.convertScaleAbs(scharr_y)
        edge_mask = cv2.addWeighted(abs_scharr_x, 0.5, abs_scharr_y, 0.5, 0)
        _, edge_mask = cv2.threshold(edge_mask, 192, 255, cv2.THRESH_BINARY)

        # Apply corner filter
        ir_float = np.float32(ir_image)
        corners = cv2.cornerHarris(ir_float, 2, 3, 0.04)
        _, harris_mask = cv2.threshold(corners, 300, 255, cv2.THRESH_BINARY)
        harris_mask_resized = cv2.resize(harris_mask, (edge_mask.shape[1], edge_mask.shape[0]))
        print("Edge Mask Shape:", edge_mask.shape)
        print("Harris Mask Resized Shape:", harris_mask_resized.shape)
        edge_mask_uint8 = np.uint8(edge_mask)
        harris_mask_resized_uint8 = np.uint8(harris_mask_resized)
        # Combine edge and corner masks
        combined_mask = cv2.bitwise_or(edge_mask_uint8, harris_mask_resized_uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # Copy masked depth values
        output_depth = np.zeros_like(downsampled_depth)
        output_depth = np.where(combined_mask != 0, downsampled_depth, 0)

        # Further processing or visualization
        cv2.imshow('Output Depth', output_depth)

        # Exit loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
