#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg_share = FindPackageShare("car_bot")


    # Gazebo + robot + controllers + RViz + twist_mux + twist_to_stamped
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [pkg_share, "launch", "gazebo_launch.py"]
            )
        ),
        launch_arguments={
            "gui": "true",
            "rviz_config": PathJoinSubstitution(
                [pkg_share, "config", "slam.rviz"]
            ),
        }.items(),
    )



    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [pkg_share, "launch", "online_async_launch.py"]
            )
        ),
        launch_arguments={
            "use_sim_time": "true",
            "autostart": "true",
        }.items(),
    )





    return LaunchDescription([

        # Start Gazebo first
        gazebo,

        # Give robot/controller time to start
        TimerAction(
            period=10.0,
            actions=[slam
                
            ]
        ),

    ])