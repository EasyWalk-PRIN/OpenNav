from datetime import datetime
import pyrealsense2 as rs
import numpy as np
from open3d.visualization import Visualizer
from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# Start streaming
profile = pipeline.start(config)

# Initialize visualization outside the loop
vis = Visualizer()
vis.create_window("Tests")
pcd = PointCloud()
vis.add_geometry(pcd)

# Streaming loop
try:
    while True:
        dt0 = datetime.now()
        pcd.clear()
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            print("No frames received.")
            continue
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        vtx = np.asarray(points.get_vertices())
        if len(vtx) == 0:
            print("No points in the point cloud data.")
            continue
        pcd.points = Vector3dVector(vtx)
        vis.update_geometry()  # Update geometry
        vis.poll_events()
        vis.update_renderer()
        process_time = datetime.now() - dt0
        print("FPS = {0}".format(1 / process_time.total_seconds()))

finally:
    pipeline.stop()
