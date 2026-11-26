#!/usr/bin/env python3
"""
Launch file for Target Visualization in car_bot autonomous robot system.

Features:
- Launches target_visualization_node (publishes target pins, labels, approach poses).
- Optionally launches RViz2 with pre-configured /target_markers MarkerArray display.
- Keeps visualization strictly optional and decoupled from core Nav2 navigation.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('location_navigation')

    default_locations_yaml = os.path.join(pkg_share, 'config', 'locations.yaml')
    default_rviz_config = os.path.join(pkg_share, 'config', 'target_visualization.rviz')
    default_map_yaml = '/home/reyan/case_study/room_map.yaml'
    default_output_dir = '/home/reyan/case_study/output'

    # Declare launch arguments
    declare_rviz = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Launch RViz2 with target visualization displays'
    )
    declare_rviz_config = DeclareLaunchArgument(
        'rviz_config',
        default_value=default_rviz_config,
        description='Path to RViz2 configuration file'
    )
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock'
    )
    declare_locations_yaml = DeclareLaunchArgument(
        'locations_yaml',
        default_value=default_locations_yaml,
        description='Path to locations.yaml file'
    )
    declare_map_yaml = DeclareLaunchArgument(
        'map_yaml',
        default_value=default_map_yaml,
        description='Path to room_map.yaml file'
    )
    declare_output_dir = DeclareLaunchArgument(
        'output_dir',
        default_value=default_output_dir,
        description='Directory to store generated PNG map visualizations'
    )

    rviz_condition = IfCondition(LaunchConfiguration('rviz'))

    # Target Visualization Node
    target_viz_node = Node(
        package='location_navigation',
        executable='target_visualization_node',
        name='target_visualization_node',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'locations_yaml': LaunchConfiguration('locations_yaml'),
            'map_yaml': LaunchConfiguration('map_yaml'),
            'output_dir': LaunchConfiguration('output_dir'),
            'publish_rate_hz': 1.0,
            'auto_generate_png': True,
        }]
    )

    # RViz2 Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_target_viz',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        condition=rviz_condition,
    )

    return LaunchDescription([
        declare_rviz,
        declare_rviz_config,
        declare_use_sim_time,
        declare_locations_yaml,
        declare_map_yaml,
        declare_output_dir,
        target_viz_node,
        rviz_node,
    ])
