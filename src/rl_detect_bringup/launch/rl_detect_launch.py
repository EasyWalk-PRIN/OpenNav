from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rl_detect',
            executable='intel_publisher_yolo_3dbbox_node2',
            name='intel_publisher_yolo_3dbbox_node2',
            output='screen',
            parameters=[
                {"segmentation": "mobile_sam.pt"}, ## segmentation model
                {"detection": "yolov8l-world.pt"}, ## detection model
                {"classes": "bottle, monitor, dustbin, cup, chair, backpack, laptop"}, ## list of classes of interests
                {'output_markers': 'True'}, ## visualization of BBOxes in RVIZ2
                {'output_pointcloud': 'True'}, ## visualization of segmented pointcloud of objects in RVIZ2
                {'output_detection_3d': 'True'} ## detection messages in ROS2
            ]
        )
    ])

