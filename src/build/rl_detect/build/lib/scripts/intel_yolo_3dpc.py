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
import math
import time
import pdb
import open3d as o3d
class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

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
        self.state = AppState()

        self.sam_model = SAM('/home/rameez/ew_rs/src/rl_detect/rl_detect/models/mobile_sam.pt')
        self.model_path = self.get_parameter('model_path').value
        self.embed_path = self.get_parameter('embed_path').value

        # Load class embeddings
        self.class_embeddings, self.class_list = read_class_embeddings(self.embed_path)

        # Initialize YOLO-World object detector
        self.yoloworld_detector = YOLOWorld(self.model_path, conf_thres=0.1, iou_thres=0.5)
        # Set up the RealSense D455 camera
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)
        self.config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        self.cfg = self.pipeline.start(self.config)
        self.drawer = DetectionDrawer(self.sam_model ,self.class_list)
        self.bridge = CvBridge()

        self.publisher = self.create_publisher(Image, 'yolo_detected_objects', 10)
        self.depth_scale = self.cfg.get_device().first_depth_sensor().get_depth_scale()

        # Initialize occupancy map using point cloud
        self.depth_intrin = None
##############
        # Get stream profile and camera intrinsics
        profile = self.pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        # Processing blocks
        self.pc = rs.pointcloud()
        self.decimate = rs.decimation_filter()
        self.decimate.set_option(rs.option.filter_magnitude, 2 ** self.state.decimate)
        self.colorizer = rs.colorizer()

        cv2.namedWindow(self.state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(self.state.WIN_NAME, w, h)
        cv2.setMouseCallback(self.state.WIN_NAME, self.mouse_cb)
        self.intr = profile.get_stream(
            rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.intr.width, self.intr.height, self.intr.fx, self.intr.fy, self.intr.ppx, self.intr.ppy)
        self.extrinsic = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        self.out = np.empty((h, w, 3), dtype=np.uint8)
##############

    def mouse_cb(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.state.mouse_btns[0] = True

        if event == cv2.EVENT_LBUTTONUP:
            self.state.mouse_btns[0] = False

        if event == cv2.EVENT_RBUTTONDOWN:
            self.state.mouse_btns[1] = True

        if event == cv2.EVENT_RBUTTONUP:
            self.state.mouse_btns[1] = False

        if event == cv2.EVENT_MBUTTONDOWN:
            self.state.mouse_btns[2] = True

        if event == cv2.EVENT_MBUTTONUP:
            self.state.mouse_btns[2] = False

        if event == cv2.EVENT_MOUSEMOVE:

            h, w = self.out.shape[:2]
            dx, dy = x - self.state.prev_mouse[0], y - self.state.prev_mouse[1]

            if self.state.mouse_btns[0]:
                self.state.yaw += float(dx) / w * 2
                self.state.pitch -= float(dy) / h * 2

            elif self.state.mouse_btns[1]:
                dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
                self.state.translation -= np.dot(self.state.rotation, dp)

            elif self.state.mouse_btns[2]:
                dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
                self.state.translation[2] += dz
                self.state.distance -= dz

        if event == cv2.EVENT_MOUSEWHEEL:
            dz = math.copysign(0.1, flags)
            self.state.translation[2] += dz
            self.state.distance -= dz

        self.state.prev_mouse = (x, y)
    def project(self,v):
        """project 3d vector array to 2d"""
        h, w = self.out.shape[:2]
        view_aspect = float(h)/w

        # ignore divide by zero for invalid depth
        with np.errstate(divide='ignore', invalid='ignore'):
            proj = v[:, :-1] / v[:, -1, np.newaxis] * \
                (w*view_aspect, h) + (w/2.0, h/2.0)

        # near clipping
        znear = 0.03
        proj[v[:, 2] < znear] = np.nan
        return proj


    def view(self, v):
        """apply view transformation on vector array"""
        return np.dot(v - self.state.pivot, self.state.rotation) + self.state.pivot - self.state.translation


    def line3d(self, out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
        """draw a 3d line from pt1 to pt2"""
        p0 = self.project(pt1.reshape(-1, 3))[0]
        p1 = self.project(pt2.reshape(-1, 3))[0]
        if np.isnan(p0).any() or np.isnan(p1).any():
            return
        p0 = tuple(p0.astype(int))
        p1 = tuple(p1.astype(int))
        rect = (0, 0, out.shape[1], out.shape[0])
        inside, p0, p1 = cv2.clipLine(rect, p0, p1)
        if inside:
            cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


    def grid(self, out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
        """draw a grid on xz plane"""
        pos = np.array(pos)
        s = size / float(n)
        s2 = 0.5 * size
        for i in range(0, n+1):
            x = -s2 + i*s
            self.line3d(out, self.view(pos + np.dot((x, 0, -s2), rotation)),
                self.view(pos + np.dot((x, 0, s2), rotation)), color)
        for i in range(0, n+1):
            z = -s2 + i*s
            self.line3d(out, self.view(pos + np.dot((-s2, 0, z), rotation)),
                self.view(pos + np.dot((s2, 0, z), rotation)), color)


    def axes(self, out, pos, rotation=np.eye(3), size=0.075, thickness=2):
        """draw 3d axes"""
        self.line3d(out, pos, pos +
            np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
        self.line3d(out, pos, pos +
            np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
        self.line3d(out, pos, pos +
            np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


    def frustum(self, out, intrinsics, color=(0x40, 0x40, 0x40)):
        """draw camera's frustum"""
        orig = self.view([0, 0, 0])
        w, h = intrinsics.width, intrinsics.height

        for d in range(1, 6, 2):
            def get_point(x, y):
                p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
                self.line3d(out, orig, self.view(p), color)
                return p

            top_left = get_point(0, 0)
            top_right = get_point(w, 0)
            bottom_right = get_point(w, h)
            bottom_left = get_point(0, h)

            self.line3d(out, self.view(top_left), self.view(top_right), color)
            self.line3d(out, self.view(top_right), self.view(bottom_right), color)
            self.line3d(out, self.view(bottom_right), self.view(bottom_left), color)
            self.line3d(out, self.view(bottom_left), self.view(top_left), color)


    def pointcloud(self, out, verts, texcoords, color, painter=True):
        """draw point cloud with optional painter's algorithm"""
        if painter:
            # Painter's algo, sort points from back to front

            # get reverse sorted indices by z (in view-space)
            # https://gist.github.com/stevenvo/e3dad127598842459b68
            v = self.view(verts)
            s = v[:, 2].argsort()[::-1]
            proj = self.project(v[s])
        else:
            proj = self.project(self.view(verts))

        if self.state.scale:
            proj *= 0.5**self.state.decimate

        h, w = out.shape[:2]

        # proj now contains 2d image coordinates
        j, i = proj.astype(np.uint32).T

        # create a mask to ignore out-of-bound indices
        im = (i >= 0) & (i < h)
        jm = (j >= 0) & (j < w)
        m = im & jm

        cw, ch = color.shape[:2][::-1]
        if painter:
            # sort texcoord with same indices as above
            # texcoords are [0..1] and relative to top-left pixel corner,
            # multiply by size and add 0.5 to center
            v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
        else:
            v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
        # clip texcoords to image
        np.clip(u, 0, ch-1, out=u)
        np.clip(v, 0, cw-1, out=v)

        # perform uv-mapping
        out[i[m], j[m]] = color[u[m], v[m]]
    
    def run(self):
        while rclpy.ok():
            if not self.state.paused:
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                depth_frame = self.decimate.process(depth_frame)

                # Grab new intrinsics (may be changed by decimation)
                depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
                w, h = depth_intrinsics.width, depth_intrinsics.height

                depth_image = np.asanyarray(depth_frame.get_data())
                o3d_depth = o3d.geometry.Image(depth_image)
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = depth_image * self.depth_scale
                boxes, scores, class_ids = self.yoloworld_detector(color_image, self.class_embeddings)
                combined_img, semantic_mask = self.drawer(color_image, depth_image, boxes, scores, class_ids)
                depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
                #change shape of semantic mask same as depth image using resize get size of depth image
                w = depth_image.shape[1]
                h = depth_image.shape[0]
                semantic_mask_color = cv2.resize(semantic_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                print(semantic_mask_color.shape)
                print(semantic_mask_color.dtype)
                o3d_color = o3d.geometry.Image(semantic_mask_color)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        o3d_color, o3d_depth, depth_scale=4000.0, convert_rgb_to_intensity=False)
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd, self.pinhole_camera_intrinsic, self.extrinsic)
                # save point cloud
                o3d.io.write_point_cloud("/home/rameez/pointcloud.ply", pcd)
                
                hull, _ = pcd.compute_convex_hull()
                hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
                hull_ls.paint_uniform_color([1, 0, 0])  # Paint hull in red

                # Create a mesh containing both the point cloud vertices and convex hull vertices
                combined_mesh = o3d.geometry.TriangleMesh()
                combined_mesh.vertices = o3d.utility.Vector3dVector(np.vstack((np.asarray(pcd.points), np.asarray(hull.vertices))))
                combined_mesh.triangles = o3d.utility.Vector3iVector(np.vstack((np.asarray(hull.triangles), np.asarray(hull.triangles) + len(pcd.points))))

                # Save the combined mesh
                o3d.io.write_triangle_mesh("/home/rameez/pointcloud_with_detections.ply", combined_mesh)
                if self.state.color:
                    self.mapped_frame, color_source = color_frame, semantic_mask
                else:
                    self.mapped_frame, color_source = depth_frame, depth_colormap

                self.points = self.pc.calculate(depth_frame)
                self.pc.map_to(self.mapped_frame)

                v, t = self.points.get_vertices(), self.points.get_texture_coordinates()
                verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
                texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

                self.render(w, h, depth_intrinsics, verts, texcoords, color_source)
                # cv2.imshow("Color Image", color_image)
                cv2.imshow("Combined Image", combined_img)

    def render(self, w, h, depth_intrinsics, verts, texcoords, color_source):
        # Render
        now = time.time()

        self.out.fill(0)

        self.grid(self.out, (0, 0.5, 1), size=1, n=10)
        self.frustum(self.out, depth_intrinsics)
        self.axes(self.out, self.view([0, 0, 0]), self.state.rotation, size=0.1, thickness=1)

        if not self.state.scale or self.out.shape[:2] == (h, w):
            self.pointcloud(self.out, verts, texcoords, color_source)
        else:
            tmp = np.zeros((h, w, 3), dtype=np.uint8)
            self.pointcloud(tmp, verts, texcoords, color_source)
            tmp = cv2.resize(
                tmp, self.out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            np.putmask(self.out, tmp > 0, tmp)

        if any(self.state.mouse_btns):
            self.axes(self.out, self.view(self.state.pivot), self.state.rotation, thickness=4)

        dt = time.time() - now

        cv2.setWindowTitle(
            self.state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
            (w, h, 1.0/dt, dt*1000, "PAUSED" if self.state.paused else ""))

        cv2.imshow(self.state.WIN_NAME, self.out)
        key = cv2.waitKey(1)

        if key == ord("r"):
            self.state.reset()

        if key == ord("p"):
            self.state.paused ^= True

        if key == ord("d"):
            self.state.decimate = (self.state.decimate + 1) % 3
            self.decimate.set_option(rs.option.filter_magnitude, 2 ** self.state.decimate)

        if key == ord("z"):
            self.state.scale ^= True

        if key == ord("c"):
            self.state.color ^= True

        if key == ord("s"):
            cv2.imwrite('./out.png', self.out)

        if key == ord("e"):
            self.points.export_to_ply('./out.ply', self.mapped_frame)
    
        # if key in (27, ord("q")) or cv2.getWindowProperty(self.state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        #     break
    #####################




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
