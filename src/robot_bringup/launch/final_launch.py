#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _resolve_workspace_root(pkg_share_dir):
    """Infer the workspace root from an installed package share directory."""
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(pkg_share_dir)))
    )


def _find_existing_path(relative_path, pkg_share_dir):
    """Return the first existing candidate path across host, workspace, and container layouts."""
    ws_root = _resolve_workspace_root(pkg_share_dir)
    candidates = [
        os.path.join(ws_root, relative_path),
        os.path.join('/workspace', relative_path),
        os.path.join(os.getcwd(), relative_path),
        os.path.join(os.path.expanduser('~/case_study'), relative_path),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def generate_launch_description():
    pkg_bringup = get_package_share_directory('robot_bringup')
    pkg_new_car = get_package_share_directory('new_car_description')
    pkg_car_bot = get_package_share_directory('car_bot')

    default_map = _find_existing_path('room_map.yaml', pkg_bringup)
    default_world = _find_existing_path(
        os.path.join('small_house', 'small_house.world'),
        pkg_bringup,
    )
    default_rviz_config = os.path.join(pkg_new_car, 'rviz', 'robot.rviz')
    nav2_params_file = os.path.join(pkg_new_car, 'config', 'nav2_params.yaml')

    map_path = LaunchConfiguration('map')
    world_path = LaunchConfiguration('world')
    use_sim_time = LaunchConfiguration('use_sim_time')
    gui = LaunchConfiguration('gui')
    rviz = LaunchConfiguration('rviz')
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

    # 1. Gazebo Harmonic + New Robot + Controllers + Bridge + RViz
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, 'launch', 'simulation.launch.py')
        ),
        launch_arguments={
            'gui': gui,
            'rviz': rviz,
            'rviz_config': rviz_config_path,
            'world': world_path,
            'use_sim_time': use_sim_time,
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
        declare_gui,
        declare_rviz,
        declare_rviz_config,

        # Start Gazebo and robot simulation
        simulation_launch,

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
