"""
# Gets the target pose (x, y, yaw).
# Converts the yaw angle into quaternion orientation.
# Waits for the Nav2 action server to become ready.
# Sends the navigation goal to Nav2.
# Monitors navigation progress, cancellation, and timeout.
# Returns True if navigation succeeds, otherwise False.
# Destroys the ROS node and shuts down ROS 2.
"""

import math
import sys
import time
import os
from typing import Optional, Tuple

try:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import PoseStamped
    from nav2_msgs.action import NavigateToPose
except ImportError:
    rclpy = None
    ActionClient = None
    Node = object
    GoalStatus = None
    PoseStamped = None
    NavigateToPose = None


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    """
    # Converts the robot's turning angle into the required orientation format.
    Convert a yaw angle (radians) to a quaternion (x, y, z, w).
    Rotation is about the Z axis only (2D navigation).
    """

    half = yaw / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


class NavigationClient(Node):
    """
    Sends navigation goals to Nav2.
    """

    ACTION_SERVER = 'navigate_to_pose'


#Initializes the navigation client and configures simulation time.
    def __init__(self, use_sim_time: Optional[bool] = None):
        parameter_overrides = []
        if use_sim_time is True:
            try:
                from rclpy.parameter import Parameter
                parameter_overrides.append(
                    Parameter('use_sim_time', Parameter.Type.BOOL, True)
                )
            except Exception:
                pass
        super().__init__('location_navigation_client', parameter_overrides=parameter_overrides)
        if use_sim_time is None:
            try:
                has_clock = False
                if hasattr(self, 'count_publishers') and self.count_publishers('/clock') > 0:
                    has_clock = True
                elif hasattr(self, 'get_topic_names_and_types'):
                    topics = [name for name, _ in self.get_topic_names_and_types()]
                    if '/clock' in topics:
                        has_clock = True
                if has_clock:
                    from rclpy.parameter import Parameter
                    self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
            except Exception:
                pass
        self._client = ActionClient(self, NavigateToPose, self.ACTION_SERVER)
        self._result_future = None
        self._goal_handle = None
        self._cancel_requested = False


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

    def server_is_ready(self) -> bool:
        """Checks whether the Nav2 action server is ready.
        """
        return self._client.server_is_ready()


    def cancel_goal(self, timeout_sec: float = 5.0) -> bool:
        """
        Request cancellation of the currently active navigation goal.

        """
        self._cancel_requested = True
        if self._goal_handle is None:
            self.get_logger().info('[NavClient] No active goal to cancel.')
            return True

        self.get_logger().info('[NavClient] Requesting cancellation of active goal...')
        cancel_future = self._goal_handle.cancel_goal_async()
        rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=timeout_sec)

        if cancel_future.done():
            self.get_logger().info('[NavClient] Goal cancellation request completed.')
            self._goal_handle = None
            return True
        else:
            self.get_logger().warn('[NavClient] Cancellation request timed out.')
            return False

    # Sends the robot to the target pose and monitors the navigation result.
    # Cancels navigation if requested or if the timeout is reached.
    def navigate_to_pose(
        self,
        x: float,
        y: float,
        yaw: float = 0.0,
        frame_id: str = 'map',
        location_name: str = '',
        timeout_sec: Optional[float] = 600.0,
    ) -> bool:

        self._cancel_requested = False
        label = location_name or f'({x:.3f}, {y:.3f})'

        # Build PoseStamped goal
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = frame_id
        goal_pose.header.stamp = self.get_clock().now().to_msg()

        goal_pose.pose.position.x = float(x)
        goal_pose.pose.position.y = float(y)
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

        # Send goal asynchronously, then spin until response
        send_future = self._client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future)

        if not send_future.done():
            self.get_logger().error(f'[NavClient] Goal send future for "{label}" did not complete.')
            return False

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(
                f'[NavClient] Goal to "{label}" was REJECTED by Nav2.')
            return False

        self._goal_handle = goal_handle
        self.get_logger().info(
            f'[NavClient] Goal to "{label}" ACCEPTED. Navigating...')

        # Wait for result with responsive spinning loop
        result_future = goal_handle.get_result_async()
        start_time = time.time()

        while rclpy.ok() and not result_future.done():
            if self._cancel_requested:
                self.get_logger().warn(f'[NavClient] Navigation to "{label}" interrupted by cancel.')
                self.cancel_goal()
                return False

            if timeout_sec is not None and (time.time() - start_time) > timeout_sec:
                self.get_logger().error(
                    f'[NavClient] Navigation to "{label}" timed out after {timeout_sec}s.'
                )

                self.cancel_goal()
                return False

            rclpy.spin_until_future_complete(self, result_future, timeout_sec=0.2)

        self._goal_handle = None

        if not result_future.done():
            self.get_logger().error(f'[NavClient] Result future did not complete for "{label}".')
            return False

        result_response = result_future.result()
        status = result_response.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(
                f'[NavClient] ✓ Navigation to "{label}" SUCCEEDED.')
            return True
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warn(
                f'[NavClient] Navigation to "{label}" was CANCELLED.')
            return False
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
    # Starts navigation to a location entered by the user.
    # Finds the location and sends the goal to Nav2.
    # Cleans up and shuts down ROS 2.
    """
    
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

    raw_target = cli_args[0]
    # Alias map for backwards compatibility
    alias_map = {
        'kitchen': 'refrigerator',
        'bedroom': 'bed',
        'tv': 'sofa',
        'dining_table': 'table',
    }
    target_name = alias_map.get(raw_target.strip().lower(), raw_target.strip().lower())

    # Load location database
    yaml_path = os.path.join(
        get_package_share_directory('location_navigation'),
        'config', 'locations.yaml')
    try:
        db = LocationDatabase(yaml_path)
        pose = db.get(target_name)
    except LocationNotFoundError as exc:
        print(str(exc))
        rclpy.shutdown()
        return
    except Exception as exc:
        print(f"Failed to load location database: {exc}")
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
