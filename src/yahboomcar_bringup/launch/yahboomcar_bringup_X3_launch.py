from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration, PythonExpression

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import os
from ament_index_python.packages import get_package_share_directory

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    if not os.environ.get("PRINTED"):
        os.environ["PRINTED"] = "1"
        print("---------------------robot_type = x3---------------------")
    urdf_tutorial_path = get_package_share_path('yahboomcar_description')
    default_model_path = urdf_tutorial_path / 'urdf/yahboomcar_X3.urdf'
    default_rviz_config_path = urdf_tutorial_path / 'rviz/yahboomcar.rviz'

    gui_arg = DeclareLaunchArgument(name='gui', default_value='false', choices=['true', 'false'],
                                    description='Flag to enable joint_state_publisher_gui')
    model_arg = DeclareLaunchArgument(name='model', default_value=str(default_model_path),
                                      description='Absolute path to robot urdf file')
    rviz_arg = DeclareLaunchArgument(name='rvizconfig', default_value=str(default_rviz_config_path),
                                     description='Absolute path to rviz config file')
    pub_odom_tf_arg = DeclareLaunchArgument('pub_odom_tf', default_value='false',
                                            description='Whether to publish the tf from the original odom to the base_footprint')

    chassis_serial_arg = DeclareLaunchArgument(
        'chassis_serial_port',
        default_value='/dev/ttyUSB0',
        description='USB serial device for Yahboom Rosmaster MCU (override e.g. /dev/myserial if you use udev symlink)',
    )

    angular_scale_arg = DeclareLaunchArgument(
        'angular_scale',
        default_value='0.5',
        description='Odometry yaw scale for base_node_X3 (set to 1.0 if already calibrated)',
    )

    use_joystick_arg = DeclareLaunchArgument(
        'use_joystick', default_value='true',
        choices=['true', 'false'],
        description='Launch joy_node and yahboom_joy_X3 (set false for autonomous patrol)',
    )

    # Mechanical-asymmetry trim. This X3 chassis curves slightly when
    # commanded pure forward motion because the four motors do not run
    # at exactly the same speed under load (per-wheel encoder rates
    # differ by ~3-5%). Compensate at the chassis-command layer:
    #
    #   trim_vy_per_vx : add (trim * vx) to commanded vy. Positive value
    #                    pushes the chassis to its LEFT, so set this
    #                    positive if the robot drifts to its RIGHT when
    #                    commanded straight forward.
    #   trim_w_per_vx  : add (trim * vx) to commanded angular.z. Use only
    #                    if there is residual yaw bias after vy trim.
    #
    # Both default to 0.0 (no compensation). Tune empirically: drive 1 m
    # forward with odom_drift_checker running, read off the delta-y /
    # delta-yaw, adjust trims, restart bringup, retest.
    trim_vy_per_vx_arg = DeclareLaunchArgument(
        'trim_vy_per_vx', default_value='0.0',
        description='vy_correction = trim * vx (positive cancels rightward drift).',
    )
    trim_w_per_vx_arg = DeclareLaunchArgument(
        'trim_w_per_vx', default_value='0.0',
        description='w_correction = trim * vx (rad/s per m/s).',
    )

    joy_xspeed_limit_arg = DeclareLaunchArgument(
        'joy_xspeed_limit', default_value='0.15',
        description='Max forward/back speed from joystick (m/s). Full stick = this value.',
    )
    joy_yspeed_limit_arg = DeclareLaunchArgument(
        'joy_yspeed_limit', default_value='0.15',
        description='Max strafe speed from joystick (m/s). Full stick = this value.',
    )
    joy_angular_speed_limit_arg = DeclareLaunchArgument(
        'joy_angular_speed_limit', default_value='0.5',
        description='Max turn rate from joystick (rad/s). Full stick = this value.',
    )

    ekf_frequency_arg = DeclareLaunchArgument(
        'ekf_frequency', default_value='10.0',
        description='EKF output rate (Hz). Default 10 on 8 GB Jetson; use 30.0 if CPU headroom.',
    )
    ekf_sensor_timeout_arg = DeclareLaunchArgument(
        'ekf_sensor_timeout', default_value='0.2',
        description='EKF sensor timeout (s). Should be ~2× (1/ekf_frequency).',
    )
    use_joint_state_publisher_arg = DeclareLaunchArgument(
        'use_joint_state_publisher', default_value='false',
        choices=['true', 'false'],
        description='Launch joint_state_publisher (driver already publishes /joint_states).',
    )
    use_ekf_arg = DeclareLaunchArgument(
        'use_ekf', default_value='true',
        choices=['true', 'false'],
        description='Launch IMU filter + EKF for /odom. Set false for lightweight SLAM (wheel odom TF only).',
    )

    robot_description = ParameterValue(Command(['xacro ', LaunchConfiguration('model')]),
                                       value_type=str)

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}]
    )

    # Depending on gui parameter, either launch joint_state_publisher or joint_state_publisher_gui
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('use_joint_state_publisher'), "' == 'true' and '",
            LaunchConfiguration('gui'), "' == 'false'"
        ])),
    )

    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        condition=IfCondition(LaunchConfiguration('gui'))
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rvizconfig')],
    )

    driver_node = Node(
        package='yahboomcar_bringup',
        executable='Mcnamu_driver_X3',
        parameters=[
            {'serial_port': ParameterValue(LaunchConfiguration('chassis_serial_port'), value_type=str)},
            {'trim_vy_per_vx': ParameterValue(LaunchConfiguration('trim_vy_per_vx'), value_type=float)},
            {'trim_w_per_vx':  ParameterValue(LaunchConfiguration('trim_w_per_vx'),  value_type=float)},
        ],
    )

    base_node = Node(
        package='yahboomcar_base_node',
        executable='base_node_X3',
        # 当使用ekf融合时，该tf有ekf发布
        parameters=[{
            'pub_odom_tf': LaunchConfiguration('pub_odom_tf'),
            'linear_scale_x': 1.0,
            'linear_scale_y': 1.0,
            'angular_scale': LaunchConfiguration('angular_scale'),
        }]
    )

    imu_filter_config = os.path.join(              
        get_package_share_directory('yahboomcar_bringup'),
        'param',
        'imu_filter_param.yaml'
    ) 

    imu_filter_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        condition=IfCondition(LaunchConfiguration('use_ekf')),
        parameters=[imu_filter_config]
    )

    ekf_params = os.path.join(
        get_package_share_directory('robot_localization'),
        'params', 'ekf_x1_x3.yaml')

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_ekf')),
        parameters=[
            ekf_params,
            {
                'frequency': ParameterValue(
                    LaunchConfiguration('ekf_frequency'), value_type=float),
                'sensor_timeout': ParameterValue(
                    LaunchConfiguration('ekf_sensor_timeout'), value_type=float),
            },
        ],
        remappings=[('/odometry/filtered', '/odom')],
    )

    yahboom_joy_node = Node(
        package='yahboomcar_ctrl',
        executable='yahboom_joy_X3',
        condition=IfCondition(LaunchConfiguration('use_joystick')),
        parameters=[{
            'xspeed_limit': ParameterValue(LaunchConfiguration('joy_xspeed_limit'), value_type=float),
            'yspeed_limit': ParameterValue(LaunchConfiguration('joy_yspeed_limit'), value_type=float),
            'angular_speed_limit': ParameterValue(LaunchConfiguration('joy_angular_speed_limit'), value_type=float),
        }],
    )
    joy_node = Node(
        package='joy',
        executable='joy_node',
        condition=IfCondition(LaunchConfiguration('use_joystick')),
    )

    return LaunchDescription([
        gui_arg,
        model_arg,
        rviz_arg,
        pub_odom_tf_arg,
        chassis_serial_arg,
        angular_scale_arg,
        use_joystick_arg,
        trim_vy_per_vx_arg,
        trim_w_per_vx_arg,
        joy_xspeed_limit_arg,
        joy_yspeed_limit_arg,
        joy_angular_speed_limit_arg,
        ekf_frequency_arg,
        ekf_sensor_timeout_arg,
        use_joint_state_publisher_arg,
        use_ekf_arg,
        joint_state_publisher_node,
        joint_state_publisher_gui_node,
        robot_state_publisher_node,
        # rviz_node
        driver_node,
        base_node,
        imu_filter_node,
        ekf_node,
        yahboom_joy_node,
        joy_node
    ])
