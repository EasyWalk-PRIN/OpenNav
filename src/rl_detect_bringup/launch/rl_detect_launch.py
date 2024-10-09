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
                {"segmentation": "mobile_sam.pt"},
                {"detection": "yolov8l-world.pt"},
                {"classes": "bottle, monitor, dustbin, cup, chair, backpack, laptop"},
                {'output_markers': 'True'},
                {'output_pointcloud': 'True'},
                {'output_detection_3d': 'True'}
            ]
        )
    ])

