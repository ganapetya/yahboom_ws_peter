"""Optional all-in-one SLAM stack — Yahboom packages only.

Manual four-terminal workflow in phase1-status.md is equally valid.
"""
import glob
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

LIDAR_BY_ID = (
    '/dev/serial/by-id/'
    'usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0'
)


def _find_chassis_port() -> str:
    for sys_path in sorted(glob.glob('/sys/class/tty/ttyUSB*')):
        dev = os.path.basename(sys_path)
        model_path = os.path.join(sys_path, 'device', 'idProduct')
        try:
            with open(model_path, 'r', encoding='ascii') as f:
                if f.read().strip() == '7523':
                    return f'/dev/{dev}'
        except OSError:
            continue
    return '/dev/ttyUSB0'


def generate_launch_description():
    pkg_bringup = get_package_share_directory('yahboomcar_bringup')
    pkg_sllidar = get_package_share_directory('sllidar_ros2')
    pkg_slam = get_package_share_directory('slam_toolbox')

    slam_params = os.path.join(pkg_bringup, 'param', 'mapper_params_online_async.yaml')
    fastdds_xml = os.path.join(pkg_bringup, 'param', 'fastdds_mapping.xml')
    chassis_default = _find_chassis_port()

    chassis_serial_arg = DeclareLaunchArgument(
        'chassis_serial_port', default_value=chassis_default)
    trim_vy_arg = DeclareLaunchArgument('trim_vy_per_vx', default_value='0.012')
    lidar_port_arg = DeclareLaunchArgument('lidar_serial_port', default_value=LIDAR_BY_ID)
    slam_delay_arg = DeclareLaunchArgument('slam_start_delay_sec', default_value='8.0')

    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, 'launch', 'yahboomcar_bringup_X3_launch.py')),
        launch_arguments={
            'chassis_serial_port': LaunchConfiguration('chassis_serial_port'),
            'use_joystick': 'true',
            'trim_vy_per_vx': LaunchConfiguration('trim_vy_per_vx'),
            'use_ekf': 'true',
            'pub_odom_tf': 'false',
            'use_joint_state_publisher': 'false',
            'ekf_frequency': '10.0',
        }.items(),
    )

    lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_sllidar, 'launch', 'sllidar_launch.py')),
        launch_arguments={
            'serial_port': LaunchConfiguration('lidar_serial_port'),
            'serial_baudrate': '115200',
            'frame_id': 'laser_link',
        }.items(),
    )

    scan_filter = Node(
        package='yahboomcar_bringup',
        executable='scan_front_filter',
        name='scan_front_filter',
        output='screen',
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_slam, 'launch', 'online_async_launch.py')),
        launch_arguments={
            'use_sim_time': 'false',
            'slam_params_file': slam_params,
        }.items(),
    )

    return LaunchDescription([
        SetEnvironmentVariable(name='FASTRTPS_DEFAULT_PROFILES_FILE', value=fastdds_xml),
        LogInfo(msg=f'SLAM mapping (yahboom stack). Chassis: {chassis_default}'),
        chassis_serial_arg,
        trim_vy_arg,
        lidar_port_arg,
        slam_delay_arg,
        bringup,
        TimerAction(period=3.0, actions=[lidar]),
        TimerAction(period=5.0, actions=[scan_filter]),
        TimerAction(period=LaunchConfiguration('slam_start_delay_sec'), actions=[slam]),
    ])
