# Copyright 2021 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "gui",
            default_value="true",
            description="Start RViz2 automatically with this launch file.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "world",
            default_value=PathJoinSubstitution(
                [FindPackageShare("car_bot"), "world", "navigation.sdf"]
            ),
            description="Full path to the world SDF file to load",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "rviz_config",
            default_value=PathJoinSubstitution(
                [FindPackageShare("car_bot"), "config", "gazebo.rviz"]
            ),
            description="load rviz config file",
        )
    )

    # Initialize arguments
    gui = LaunchConfiguration("gui")
    world = LaunchConfiguration("world")

    # Gazebo with GUI
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
        ),
        launch_arguments=[("gz_args", ["-r -v 3 ", world])],
        condition=IfCondition(gui),
    )

    # Gazebo headless
    gazebo_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
        ),
        launch_arguments=[("gz_args", ["--headless-rendering -s -r -v 3 ", world])],
        condition=UnlessCondition(gui),
    )

    # Bridge params
    bridge_params = os.path.join(
        get_package_share_directory('car_bot'), 'config', 'gz_bridge.yaml'
    )

    # Gazebo bridge
    gazebo_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=['--ros-args',
                   '-p',
                   f'config_file:={bridge_params}'],
    )

    # Spawn entity from /robot_description
    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic",
            "/robot_description",
            "-name",
            "rrbot_system_position",
            "-allow_renaming",
            "true",
            "-x",
            "-1.0",
            "-y",
            "0.0",
            "-z",
            "0.2",
        ],
    )

    # Robot description
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("car_bot"), "urdf", "gazebo.urdf.xacro"]
            ),
            " ",
            "use_gazebo:=true",
        ]
    )
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    # Controllers
    robot_controllers = PathJoinSubstitution(
        [FindPackageShare("car_bot"), "config", "control.yaml"]
    )

    twist_mux_params = PathJoinSubstitution(
    [FindPackageShare("car_bot"), "config", "twist_mux.yaml"]
    )
    # RViz config
    #rviz_config_file = PathJoinSubstitution(
    #    [FindPackageShare("car_bot"), "config", "gazebo.rviz"]
    #)

    node_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    joint_state_broadcaster_spawner = TimerAction(
        period=5.0,
        actions=[Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster"],
            output="screen",
        )]
    )

    robot_controller_spawner = TimerAction(
        period=6.0,
        actions=[Node(
            package="controller_manager",
            executable="spawner",
            arguments=["diffbot_base_controller", "--param-file", robot_controllers],
            output="screen",
        )]
    )

    rviz_node = TimerAction(
        period=1.0,
        actions=[Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="log",
            arguments=['-d', LaunchConfiguration('rviz_config')],
            condition=IfCondition(gui),
        )]
    )

    twist_to_stamped = Node(
        package='car_bot',
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
            ('/cmd_vel_out', 'diffbot_base_controller/cmd_vel')
        ]
    )




    nodes = [
        gazebo,
        gazebo_headless,
        gazebo_bridge,
        node_robot_state_publisher,
        gz_spawn_entity,
        joint_state_broadcaster_spawner,
        robot_controller_spawner,
        rviz_node,
        twist_to_stamped,
        twist_mux,
        
    ]

    return LaunchDescription(declared_arguments + nodes)
