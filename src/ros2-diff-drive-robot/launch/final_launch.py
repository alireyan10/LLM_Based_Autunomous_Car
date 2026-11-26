#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg_car_bot = FindPackageShare("car_bot")
    pkg_new_car = FindPackageShare("new_car_description")

    use_new_car = LaunchConfiguration("use_new_car")
    declare_use_new_car = DeclareLaunchArgument(
        "use_new_car",
        default_value="true",
        description="Whether to use the new car model (new_car_description) or legacy robot",
    )

    map_arg = LaunchConfiguration("map")
    default_map = "/workspace/room_map.yaml"
    if not os.path.exists(default_map) and os.path.exists("/home/reyan/case_study/room_map.yaml"):
        default_map = "/home/reyan/case_study/room_map.yaml"

    declare_map = DeclareLaunchArgument(
        "map",
        default_value=default_map,
        description="Full path to map yaml file",
    )

    rviz_config_arg = LaunchConfiguration("rviz_config")
    declare_rviz_config = DeclareLaunchArgument(
        "rviz_config",
        default_value=PathJoinSubstitution(
            [pkg_new_car, "rviz", "robot.rviz"]
        ),
        description="Full path to RViz configuration file",
    )

    # 1a. New car Gazebo launch
    gazebo_new_car = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [pkg_new_car, "launch", "gazebo.launch.py"]
            )
        ),
        launch_arguments={
            "gui": "true",
            "rviz": "true",
            "rviz_config": rviz_config_arg,
        }.items(),
        condition=IfCondition(use_new_car),
    )

    # 1b. Legacy robot Gazebo launch
    gazebo_legacy = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [pkg_car_bot, "launch", "gazebo_launch.py"]
            )
        ),
        launch_arguments={
            "gui": "true",
            "rviz_config": rviz_config_arg,
        }.items(),
        condition=UnlessCondition(use_new_car),
    )

    # 2. Localization (map_server + AMCL)
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [pkg_car_bot, "launch", "localization_launch.py"]
            )
        ),
        launch_arguments={
            "use_sim_time": "true",
            "autostart": "true",
            "map": map_arg,
        }.items(),
    )

    # 3. Navigation (planner/controller/bt navigator)
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [pkg_car_bot, "launch", "navigation_launch.py"]
            )
        ),
        launch_arguments={
            "use_sim_time": "true",
            "autostart": "true",
            "params_file": PathJoinSubstitution(
                [pkg_car_bot, "config", "nav2_params.yaml"]
            ),
        }.items(),
    )

    return LaunchDescription([
        declare_use_new_car,
        declare_map,
        declare_rviz_config,

        # Start Gazebo first (new car or legacy)
        gazebo_new_car,
        gazebo_legacy,

        # Give robot/controller time to start
        TimerAction(
            period=10.0,
            actions=[
                localization
            ]
        ),

        # Start Nav2 after AMCL/map
        TimerAction(
            period=15.0,
            actions=[
                navigation
            ]
        ),
    ])
