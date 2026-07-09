from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value='/home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/src/patrol_manager/config/patrol_manager_params.yaml',
        description='Full path to patrol manager params YAML',
    )

    patrol_node = Node(
        package='patrol_manager',
        executable='patrol_manager_node',
        name='patrol_manager',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
    )

    return LaunchDescription([
        params_file_arg,
        patrol_node,
    ])
