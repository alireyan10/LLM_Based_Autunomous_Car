import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('new_car_description')
    gz_share = get_package_share_directory('ros_gz_sim')

    # Default world path (supports both docker /workspace and host path)
    default_world = '/workspace/small_house/small_house.world'
    if not os.path.exists(default_world) and os.path.exists('/home/reyan/case_study/small_house/small_house.world'):
        default_world = '/home/reyan/case_study/small_house/small_house.world'

    xacro_file = os.path.join(pkg_share, 'urdf', 'robot.urdf.xacro')
    bridge_config = os.path.join(pkg_share, 'config', 'bridge.yaml')
    gazebo_launch_file = os.path.join(gz_share, 'launch', 'gz_sim.launch.py')
    default_rviz_config = os.path.join(pkg_share, 'rviz', 'robot.rviz')

    # Launch Configurations
    world = LaunchConfiguration('world')
    gui = LaunchConfiguration('gui')
    rviz = LaunchConfiguration('rviz')
    rviz_config = LaunchConfiguration('rviz_config')
    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')
    spawn_z = LaunchConfiguration('spawn_z')
    spawn_yaw = LaunchConfiguration('spawn_yaw')

    declare_world = DeclareLaunchArgument(
        'world',
        default_value=default_world,
        description='Full path to world SDF/world file'
    )

    declare_gui = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Whether to launch Gazebo GUI'
    )

    declare_rviz = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Whether to launch RViz'
    )

    declare_rviz_config = DeclareLaunchArgument(
        'rviz_config',
        default_value=default_rviz_config,
        description='Full path to RViz configuration file'
    )

    declare_spawn_x = DeclareLaunchArgument(
        'spawn_x',
        default_value='-1.0',
        description='X coordinate for robot spawn'
    )

    declare_spawn_y = DeclareLaunchArgument(
        'spawn_y',
        default_value='0.0',
        description='Y coordinate for robot spawn'
    )

    declare_spawn_z = DeclareLaunchArgument(
        'spawn_z',
        default_value='0.03',
        description='Z coordinate for robot spawn (3 cm clearance above floor to prevent penetration)'
    )

    declare_spawn_yaw = DeclareLaunchArgument(
        'spawn_yaw',
        default_value='0.0',
        description='Yaw rotation for robot spawn'
    )

    # Gazebo Harmonic resource path for models, static house, and package meshes
    resource_paths = [
        '/workspace/small_house/models_static_test',
        '/workspace/small_house',
        '/workspace/small_house/maps',
        '/home/reyan/case_study/small_house/models_static_test',
        '/home/reyan/case_study/small_house',
        '/home/reyan/case_study/small_house/maps',
        os.path.dirname(pkg_share),
        pkg_share
    ]
    set_gz_resource_path = AppendEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=':'.join(p for p in resource_paths if os.path.exists(p))
    )

    # Robot Description via Xacro
    robot_description = ParameterValue(
        Command([
            FindExecutable(name='xacro'),
            ' ',
            xacro_file
        ]),
        value_type=str
    )

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True
        }],
        output='screen'
    )

    # Gazebo Harmonic Sim Launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file),
        launch_arguments={
            'gz_args': ['-r ', world]
        }.items()
    )

    # ROS-GZ Bridge (Clock & LiDAR /scan)
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '--ros-args',
            '-p',
            f'config_file:={bridge_config}'
        ],
        output='screen'
    )

    # Spawn Robot Entity in Gazebo Harmonic
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'new_car',
            '-topic', '/robot_description',
            '-x', spawn_x,
            '-y', spawn_y,
            '-z', spawn_z,
            '-Y', spawn_yaw
        ],
        output='screen'
    )

    # Controller Spawners
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    diff_drive_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['diff_drive_controller', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    # Controller activation after robot spawn
    load_joint_state_broadcaster = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_robot,
            on_exit=[joint_state_broadcaster_spawner]
        )
    )

    load_diff_drive_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[diff_drive_controller_spawner]
        )
    )

    # RViz Node loading configured rviz_config with simulation time
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(rviz),
        output='screen'
    )

    # Velocity conversion and multiplexing
    twist_mux_params = os.path.join(pkg_share, 'config', 'twist_mux.yaml')

    twist_to_stamped = Node(
        package='new_car_description',
        executable='twist_to_stamped',
        name='twist_to_stamped',
        output='screen',
        parameters=[{
            'input_topic': '/cmd_vel_nav',
            'output_topic': '/cmd_vel_nav_stamped',
            'frame_id': 'base_link',
            'use_sim_time': True,
        }]
    )

    twist_mux = Node(
        package='twist_mux',
        executable='twist_mux',
        name='twist_mux',
        output='screen',
        parameters=[twist_mux_params],
        remappings=[
            ('/cmd_vel_out', '/diff_drive_controller/cmd_vel')
        ]
    )

    return LaunchDescription([
        declare_world,
        declare_gui,
        declare_rviz,
        declare_rviz_config,
        declare_spawn_x,
        declare_spawn_y,
        declare_spawn_z,
        declare_spawn_yaw,
        set_gz_resource_path,
        gazebo,
        gz_bridge,
        robot_state_publisher,
        TimerAction(
            period=3.0,
            actions=[spawn_robot]
        ),
        load_joint_state_broadcaster,
        load_diff_drive_controller,
        twist_to_stamped,
        twist_mux,
        rviz_node
    ])
