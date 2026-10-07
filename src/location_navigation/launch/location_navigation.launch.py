import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    """
    Launch file for the location_navigation system.
    This does NOT start Nav2 or Gazebo. It only exposes the node if we wanted
    it to run as a service/action server continuously.
    For this version, llm_to_nav is a CLI tool, so a launch file is optional 
    but provided for completeness if we want to run a continuous service later.
    """
    
    # We will just run an empty launch description for now, as llm_to_nav 
    # is designed as a standalone command line invocation based on the request:
    # ros2 run location_navigation llm_to_nav '{"intent":"MOVE_TO_TARGET","target":"kitchen"}'
    
    return LaunchDescription([
        # Add future continuous nodes here if needed
    ])
