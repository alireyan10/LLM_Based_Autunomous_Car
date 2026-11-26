#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_new_car = get_package_share_directory('new_car_description')
    pkg_car_bot = get_package_share_directory('car_bot')

    default_map = '/workspace/room_map.yaml'
    if not os.path.exists(default_map) and os.path.exists('/home/reyan/case_study/room_map.yaml'):
        default_map = '/home/reyan/case_study/room_map.yaml'

    default_world = '/workspace/small_house/small_house.world'
    if not os.path.exists(default_world) and os.path.exists('/home/reyan/case_study/small_house/small_house.world'):
        default_world = '/home/reyan/case_study/small_house/small_house.world'

    default_rviz_config = os.path.join(pkg_new_car, 'rviz', 'robot.rviz')

    map_path = LaunchConfiguration('map')
    world_path = LaunchConfiguration('world')
    use_sim_time = LaunchConfiguration('use_sim_time')
    rviz_config_path = LaunchConfiguration('rviz_config')

    declare_map = DeclareLaunchArgument(
        'map',
        default_value=default_map,
        description='Full path to map yaml file'
    )

    declare_world = DeclareLaunchArgument(
        'world',
        default_value=default_world,
        description='Full path to world SDF file'
    )

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock'
    )

    declare_rviz_config = DeclareLaunchArgument(
        'rviz_config',
        default_value=default_rviz_config,
        description='Full path to RViz configuration file'
    )

    # 1. Gazebo Harmonic + New Robot + Controllers + Bridge + RViz
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_new_car, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'gui': 'true',
            'rviz': 'true',
            'rviz_config': rviz_config_path,
            'world': world_path,
        }.items()
    )

    # 2. Localization (map_server + AMCL)
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_car_bot, 'launch', 'localization_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'autostart': 'true',
            'map': map_path,
        }.items()
    )

    # 3. Navigation (Nav2 planner/controller/bt_navigator with tuned new_car params)
    nav2_params_file = os.path.join(pkg_new_car, 'config', 'nav2_params.yaml')
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_car_bot, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'autostart': 'true',
            'params_file': nav2_params_file,
        }.items()
    )

    return LaunchDescription([
        declare_map,
        declare_world,
        declare_use_sim_time,
        declare_rviz_config,

        # Start Gazebo and robot
        gazebo_launch,

        # Start AMCL localization after robot is initialized
        TimerAction(
            period=10.0,
            actions=[localization_launch]
        ),

        # Start Nav2 navigation stack after AMCL has localized
        TimerAction(
            period=15.0,
            actions=[navigation_launch]
        ),
    ])
