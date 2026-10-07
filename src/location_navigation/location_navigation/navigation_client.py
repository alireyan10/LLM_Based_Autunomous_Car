"""
navigation_client.py
====================
ROS 2 Nav2 action client that sends a NavigateToPose goal from a pose dict.

Designed to be imported by llm_to_nav.py, but also runnable standalone:

    ros2 run location_navigation navigation_client kitchen

The standalone mode resolves the name from the installed locations.yaml.
"""

import math
import sys
import time
import os

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose


# ---------------------------------------------------------------------------
# Helper: yaw → quaternion (rotation about Z only)
# ---------------------------------------------------------------------------
def yaw_to_quaternion(yaw: float) -> tuple:
    """
    Convert a yaw angle (radians) to a quaternion (x, y, z, w).
    Rotation is about the Z axis only (2D navigation).
    """
    half = yaw / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


# ---------------------------------------------------------------------------
# NavigationClient node
# ---------------------------------------------------------------------------
class NavigationClient(Node):
    """
    Thin ROS 2 node that wraps the nav2_msgs/NavigateToPose action client.

    Call :py:meth:`navigate_to_pose` to send a single goal and block until
    the result is available.
    """

    ACTION_SERVER = 'navigate_to_pose'

    def __init__(self):
        super().__init__('location_navigation_client')
        self._client = ActionClient(self, NavigateToPose, self.ACTION_SERVER)
        self._result_future = None
        self._goal_handle = None

    # ------------------------------------------------------------------
    def wait_for_server(self, timeout_sec: float = 10.0) -> bool:
        """Block until the Nav2 action server is available."""
        self.get_logger().info(
            f'Waiting for Nav2 action server "{self.ACTION_SERVER}" ...')
        ready = self._client.wait_for_server(timeout_sec=timeout_sec)
        if ready:
            self.get_logger().info('Nav2 action server is ready.')
        else:
            self.get_logger().error(
                f'Nav2 action server NOT available after {timeout_sec}s. '
                'Is Nav2 running?')
        return ready

    # ------------------------------------------------------------------
    def navigate_to_pose(
        self,
        x: float,
        y: float,
        yaw: float = 0.0,
        frame_id: str = 'map',
        location_name: str = '',
    ) -> bool:
        """
        Send a NavigateToPose goal and block until completion.

        Parameters
        ----------
        x, y      : target position in *frame_id*
        yaw       : target heading in radians
        frame_id  : coordinate frame (default: 'map')
        location_name : human-readable label for logging

        Returns
        -------
        True if navigation succeeded, False otherwise.
        """
        label = location_name or f'({x:.3f}, {y:.3f})'

        # Build PoseStamped goal
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = frame_id
        goal_pose.header.stamp = self.get_clock().now().to_msg()

        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.position.z = 0.0

        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        goal_pose.pose.orientation.x = qx
        goal_pose.pose.orientation.y = qy
        goal_pose.pose.orientation.z = qz
        goal_pose.pose.orientation.w = qw

        # Build action goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        self.get_logger().info(
            f'[NavClient] Sending goal → {label}  '
            f'x={x:.4f}  y={y:.4f}  yaw={yaw:.4f}  frame={frame_id}')

        # Send goal asynchronously, then spin until result
        send_future = self._client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future)

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(
                f'[NavClient] Goal to "{label}" was REJECTED by Nav2.')
            return False

        self.get_logger().info(
            f'[NavClient] Goal to "{label}" ACCEPTED. Navigating...')

        # Wait for result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result_response = result_future.result()
        status = result_response.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(
                f'[NavClient] ✓ Navigation to "{label}" SUCCEEDED.')
            return True
        else:
            self.get_logger().error(
                f'[NavClient] ✗ Navigation to "{label}" FAILED '
                f'(status={status}).')
            return False


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
def main(args=None):
    """
    Standalone test:  ros2 run location_navigation navigation_client kitchen
    Resolves the location name from installed locations.yaml and navigates.
    """
    import os
    from ament_index_python.packages import get_package_share_directory
    from location_navigation.location_database import LocationDatabase, LocationNotFoundError

    rclpy.init(args=args)

    # Parse CLI argument
    cli_args = sys.argv[1:]
    if not cli_args:
        print('Usage: ros2 run location_navigation navigation_client <location_name>')
        print('Example: ros2 run location_navigation navigation_client kitchen')
        rclpy.shutdown()
        return

    target_name = cli_args[0]

    # Load location database
    yaml_path = os.path.join(
        get_package_share_directory('location_navigation'),
        'config', 'locations.yaml')
    db = LocationDatabase(yaml_path)

    try:
        pose = db.get(target_name)
    except LocationNotFoundError as exc:
        print(str(exc))
        rclpy.shutdown()
        return

    node = NavigationClient()

    if not node.wait_for_server(timeout_sec=15.0):
        node.destroy_node()
        rclpy.shutdown()
        return

    success = node.navigate_to_pose(
        x=pose['x'],
        y=pose['y'],
        yaw=pose['yaw'],
        frame_id=pose['frame_id'],
        location_name=target_name,
    )

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
