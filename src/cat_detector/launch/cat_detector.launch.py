from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value='/home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/src/cat_detector/config/cat_detector_params.yaml',
        description='Full path to cat_detector params YAML',
    )

    detector_node = Node(
        package='cat_detector',
        executable='detector_node',
        name='cat_detector',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
    )

    return LaunchDescription([
        params_file_arg,
        detector_node,
    ])
