#!/usr/bin/env python3
"""
llm_to_nav.py
=============
Entry point for parsing LLM JSON output and sending the robot to the requested location.
Takes a JSON string argument, parses it, looks up the target location, and navigates.

Usage:
  ros2 run location_navigation llm_to_nav '{"intent":"MOVE_TO_TARGET","target":"kitchen"}'
"""

import sys
import json
import rclpy
import os
from ament_index_python.packages import get_package_share_directory

from location_navigation.location_database import LocationDatabase, LocationNotFoundError
from location_navigation.navigation_client import NavigationClient


def main(args=None):
    rclpy.init(args=args)

    # 1. Parse CLI arguments
    cli_args = sys.argv[1:]
    if not cli_args:
        print("Usage: ros2 run location_navigation llm_to_nav '<json_string>'")
        print("Example: ros2 run location_navigation llm_to_nav '{\"intent\":\"MOVE_TO_TARGET\",\"target\":\"kitchen\"}'")
        rclpy.shutdown()
        return

    json_str = cli_args[0]
    
    # 2. Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        rclpy.shutdown()
        return
        
    intent = data.get("intent", "")
    target = data.get("target", "")
    
    if intent != "MOVE_TO_TARGET":
        print(f"Ignored intent '{intent}'. Expected 'MOVE_TO_TARGET'.")
        rclpy.shutdown()
        return
        
    if not target:
        print("No 'target' specified in JSON.")
        rclpy.shutdown()
        return

    print(f"[llm_to_nav] Parsed request to move to: {target}")

    # 3. Lookup Location
    yaml_path = os.path.join(
        get_package_share_directory('location_navigation'),
        'config', 'locations.yaml')
        
    try:
        db = LocationDatabase(yaml_path)
        pose = db.get(target)
    except LocationNotFoundError as exc:
        print(str(exc))
        rclpy.shutdown()
        return
    except Exception as e:
        print(f"Failed to load location database: {e}")
        rclpy.shutdown()
        return

    # 4. Navigate
    nav_client = NavigationClient()

    if not nav_client.wait_for_server(timeout_sec=15.0):
        nav_client.destroy_node()
        rclpy.shutdown()
        return

    success = nav_client.navigate_to_pose(
        x=pose['x'],
        y=pose['y'],
        yaw=pose['yaw'],
        frame_id=pose['frame_id'],
        location_name=target,
    )

    if success:
        print(f"[llm_to_nav] Successfully reached {target}!")
    else:
        print(f"[llm_to_nav] Failed to reach {target}.")

    nav_client.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
