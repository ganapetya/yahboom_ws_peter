from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value='/home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/src/patrol_manager/config/patrol_manager_params.yaml',
        description='Full path to patrol manager params YAML',
    )
    require_audio_ready_arg = DeclareLaunchArgument(
        'require_audio_ready',
        default_value='false',
        choices=['true', 'false'],
        description='If true, hold WAKING until /cat_patrol/audio_ready (BT ready-chime).',
    )

    patrol_node = Node(
        package='patrol_manager',
        executable='patrol_manager_node',
        name='patrol_manager',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'require_audio_ready': ParameterValue(
                    LaunchConfiguration('require_audio_ready'), value_type=bool),
            },
        ],
    )

    return LaunchDescription([
        params_file_arg,
        require_audio_ready_arg,
        patrol_node,
    ])
