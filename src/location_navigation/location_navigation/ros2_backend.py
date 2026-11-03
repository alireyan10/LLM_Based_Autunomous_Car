#!/usr/bin/env python3
"""
This file connects the Sequential Executor with ROS 2 and Nav2.
It uses Nav2 to move the robot to locations, rotate, and move forward.
If DriveOnHeading is unavailable, it uses an alternative method to move forward.
It checks speed, distance, and time limits before executing actions.
It cancels ongoing actions when STOP is received, but it does not provide a hardware emergency stop.
"""

import math
import os
import sys
import time
import threading
from typing import Any, Dict, Optional, Tuple, List

try:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
    from rclpy.callback_groups import ReentrantCallbackGroup
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import PoseStamped, Point
    from nav2_msgs.action import NavigateToPose
    from builtin_interfaces.msg import Duration
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    rclpy = None
    ActionClient = None
    Node = None
    MultiThreadedExecutor = None
    SingleThreadedExecutor = None
    ReentrantCallbackGroup = None
    GoalStatus = None
    PoseStamped = None
    Point = None
    NavigateToPose = None
    Duration = None


def detect_use_sim_time(node: Optional[Any] = None) -> bool:
    """
    Determine if simulated time should be enabled.
    1. Check CLI args / sys.argv for use_sim_time:=...
    2. Check USE_SIM_TIME environment variable
    3. Check active ROS graph for /clock publishers
    4. Default to False
    """
    for arg in sys.argv:
        if "use_sim_time:=" in arg:
            val = arg.split("use_sim_time:=")[-1].strip().lower()
            return val in ("1", "true", "yes")

    if "USE_SIM_TIME" in os.environ:
        return os.environ["USE_SIM_TIME"].strip().lower() in ("1", "true", "yes")

    if node is not None and hasattr(node, "get_parameter"):
        try:
            param = node.get_parameter("use_sim_time")
            if param is not None and hasattr(param, "value") and param.value is True:
                return True
        except Exception:
            pass

    if node is not None and hasattr(node, "count_publishers"):
        try:
            if node.count_publishers("/clock") > 0:
                return True
        except Exception:
            pass

    if node is not None and hasattr(node, "get_topic_names_and_types"):
        try:
            topics = [name for name, _ in node.get_topic_names_and_types()]
            if "/clock" in topics:
                return True
        except Exception:
            pass

    return False


try:
    from nav2_msgs.action import Spin
    HAS_SPIN = True
except ImportError:
    Spin = None
    HAS_SPIN = False

try:
    from nav2_msgs.action import DriveOnHeading
    HAS_DRIVE_ON_HEADING = True
except ImportError:
    DriveOnHeading = None
    HAS_DRIVE_ON_HEADING = False

try:
    import tf2_ros
    from tf2_ros import Buffer, TransformListener
    HAS_TF2 = True
except ImportError:
    tf2_ros = None
    Buffer = None
    TransformListener = None
    HAS_TF2 = False

from location_navigation.sequential_executor import ActionExecutionBackend
from location_navigation.location_database import LocationDatabase, LocationNotFoundError


# ---------------------------------------------------------------------------
# Safety Limits & Defaults
# ---------------------------------------------------------------------------
DEFAULT_LINEAR_SPEED: float = 0.15          # m/s (safe bounded linear velocity)
MAX_SAFE_LINEAR_SPEED: float = 0.20         # m/s (maximum limit)
DEFAULT_ANGULAR_SPEED: float = 0.30         # rad/s (~17 deg/s)
MAX_SAFE_ANGULAR_SPEED: float = 0.50        # rad/s (~29 deg/s, maximum limit)

DEFAULT_FORWARD_DISTANCE: float = 1.0       # meters
DEFAULT_BACKWARD_DISTANCE: float = 1.0      # meters
DEFAULT_TURN_ANGLE_DEG: float = 90.0        # degrees

MAX_FORWARD_DISTANCE: float = 2.0           # meters (maximum limit per movement action)
MAX_MOVEMENT_DISTANCE: float = 2.0          # meters
MIN_FORWARD_DISTANCE: float = 0.01          # meters
MAX_TURN_ANGLE_DEG: float = 180.0           # degrees (maximum limit absolute turn)
DEFAULT_SERVER_TIMEOUT_SEC: float = 15.0    # seconds
ACTION_TIMEOUT_MARGIN_SEC: float = 15.0     # seconds added to theoretical duration
DEFAULT_FALLBACK_TIMEOUT_SEC: float = 120.0 # seconds for NavigateToPose execution/fallback

# Legacy room-to-target alias mapping for backward compatibility
LEGACY_TARGET_ALIASES: Dict[str, str] = {
    "kitchen": "refrigerator",
    "bedroom": "bed",
    "tv": "sofa",

    "dining_table": "table",
}


def normalize_yaw(yaw: float) -> float:
    """Normalize a yaw angle in radians to the standard range [-pi, pi]."""
    return math.atan2(math.sin(yaw), math.cos(yaw))


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    """Convert a yaw angle in radians to a quaternion tuple (x, y, z, w)."""
    half = normalize_yaw(yaw) / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


# ---------------------------------------------------------------------------
# Lightweight message structures when running in mock/isolated test mode
# ---------------------------------------------------------------------------
class _MockPose:
    def __init__(self):
        self.position = type("Position", (), {"x": 0.0, "y": 0.0, "z": 0.0})()
        self.orientation = type("Orientation", (), {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})()

class _MockPoseStamped:
    def __init__(self):
        self.header = type("Header", (), {"frame_id": "map", "stamp": None})()
        self.pose = _MockPose()

class _MockPoint:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

class _MockDuration:
    def __init__(self, sec=0, nanosec=0):
        self.sec = int(sec)
        self.nanosec = int(nanosec)

class _MockGoal:
    def __init__(self):
        self.pose = None
        self.target = None
        self.speed = 0.0
        self.target_yaw = 0.0
        self.time_allowance = None


# ---------------------------------------------------------------------------
# Ros2ActionExecutionBackend
# ---------------------------------------------------------------------------
class Ros2ActionExecutionBackend(ActionExecutionBackend):
    """
    ROS 2 Jazzy and Nav2 execution backend for SequentialActionExecutor.

    Connects validated high-level actions to the actual Nav2 action servers:
    - MOVE_TO_TARGET -> /navigate_to_pose (NavigateToPose)
    - MOVE_FORWARD   -> /drive_on_heading (DriveOnHeading) [Preferred]
                        Fallback: NavigateToPose in 'map' frame via TF current pose
    - TURN_LEFT      -> /spin (Spin, positive yaw) [Preferred]
                        Fallback: NavigateToPose in 'map' frame via TF current pose
    - TURN_RIGHT     -> /spin (Spin, negative yaw) [Preferred]
                        Fallback: NavigateToPose in 'map' frame via TF current pose
    - STOP           -> active goal cancellation on Nav2 action servers
    """

    NAV_ACTION = "navigate_to_pose"
    SPIN_ACTION = "spin"
    DRIVE_ACTION = "drive_on_heading"

    def __init__(
        self,
        node: Optional[Any] = None,
        location_database: Optional[LocationDatabase] = None,
        locations_yaml_path: Optional[str] = None,
        linear_speed: float = DEFAULT_LINEAR_SPEED,
        angular_speed: float = DEFAULT_ANGULAR_SPEED,
        max_distance: float = MAX_FORWARD_DISTANCE,
        server_timeout_sec: float = DEFAULT_SERVER_TIMEOUT_SEC,
        fallback_timeout_sec: float = DEFAULT_FALLBACK_TIMEOUT_SEC,
        stuck_timeout_sec: Optional[float] = None,
        max_timeout_sec: Optional[float] = None,
        logger: Optional[Any] = None,
        use_sim_time: Optional[bool] = None,
    ) -> None:
        """
        Initialize the ROS 2 Nav2 execution backend.
        """
        self._linear_speed = min(max(linear_speed, 0.05), MAX_SAFE_LINEAR_SPEED)
        self._angular_speed = min(max(angular_speed, 0.05), MAX_SAFE_ANGULAR_SPEED)
        self._max_distance = min(max_distance, MAX_MOVEMENT_DISTANCE)
        self._server_timeout_sec = max(server_timeout_sec, 1.0)
        self._fallback_timeout_sec = max(float(fallback_timeout_sec), 1.0)

        if stuck_timeout_sec is not None:
            self._stuck_timeout_sec = max(float(stuck_timeout_sec), 1.0)
        else:
            self._stuck_timeout_sec = float(os.getenv("NAV2_STUCK_TIMEOUT_SEC", "30.0"))

        if max_timeout_sec is not None:
            self._max_timeout_sec = max(float(max_timeout_sec), 10.0)
        else:
            self._max_timeout_sec = float(os.getenv("NAV2_MAX_TIMEOUT_SEC", "600.0"))

        # ROS 2 node setup
        self._owns_node = False
        if node is not None:
            self._node = node
        elif HAS_ROS2 and rclpy is not None and rclpy.ok():
            sim_time = use_sim_time if use_sim_time is not None else detect_use_sim_time()
            try:
                from rclpy.parameter import Parameter
                overrides = [Parameter("use_sim_time", Parameter.Type.BOOL, sim_time)]
                self._node = Node("sequential_execution_backend", parameter_overrides=overrides)
            except Exception:
                self._node = Node("sequential_execution_backend")
            self._owns_node = True
        else:
            self._node = None

        self._logger = logger or (self._node.get_logger() if self._node is not None and hasattr(self._node, "get_logger") else None)

        # Apply use_sim_time explicitly if requested or detected
        if self._node is not None:
            if use_sim_time is not None:
                self._apply_use_sim_time(use_sim_time)
            elif detect_use_sim_time(self._node):
                self._apply_use_sim_time(True)

            if hasattr(self._node, "get_parameter"):
                try:
                    if hasattr(self._node, "has_parameter") and self._node.has_parameter("fallback_timeout_sec") is True:
                        p = self._node.get_parameter("fallback_timeout_sec")
                        if p is not None and hasattr(p, "value"):
                            val = p.value
                            if isinstance(val, (int, float)) and not isinstance(val, bool):
                                self._fallback_timeout_sec = max(float(val), 1.0)
                            elif isinstance(val, str):
                                try:
                                    self._fallback_timeout_sec = max(float(val), 1.0)
                                except ValueError:
                                    pass
                    elif hasattr(self._node, "declare_parameter"):
                        self._node.declare_parameter("fallback_timeout_sec", self._fallback_timeout_sec)
                except Exception:
                    pass

        # Location Database setup
        if location_database is not None:
            self._db = location_database
        elif locations_yaml_path is not None:
            self._db = LocationDatabase(locations_yaml_path)
        else:
            self._db = self._load_default_database()

        # Action clients
        self._nav_client: Optional[Any] = None
        self._spin_client: Optional[Any] = None
        self._drive_client: Optional[Any] = None

        # Dedicated ReentrantCallbackGroup for action clients to prevent callback starvation
        self._action_cb_group = (
            ReentrantCallbackGroup()
            if (ReentrantCallbackGroup is not None and self._node is not None)
            else None
        )

        # TF Buffer and Listener for resolving current robot pose in 'map' frame
        self._tf_buffer: Optional[Any] = None
        self._tf_listener: Optional[Any] = None

        # Tracking active goal handle for cancellation
        self._active_goal_handle: Optional[Any] = None
        self._cancel_requested: bool = False
        self._last_goal_status: Optional[int] = None

        # Background executor and thread for continuous TF and action callback processing
        self._executor: Optional[Any] = None
        self._executor_thread: Optional[threading.Thread] = None

        if self._node is not None:
            self._init_action_clients()
            self._init_tf()
            self._start_spinning()

    def _apply_use_sim_time(self, use_sim: bool) -> None:
        """Explicitly declare or set use_sim_time parameter on node."""
        if self._node is None or not hasattr(self._node, "set_parameters"):
            return
        try:
            from rclpy.parameter import Parameter
            self._node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, use_sim)])
            self._log_info(f"Set use_sim_time={use_sim} on node.")
        except Exception as exc:
            self._log_warn(f"Could not set use_sim_time on node: {exc}")

    def _start_spinning(self) -> None:
        """Start a background thread spinning MultiThreadedExecutor for TF and action events."""
        if self._node is None:
            return

        # If node already attached to an executor, do not attach a second one
        if hasattr(self._node, "executor") and self._node.executor is not None:
            self._log_info("Node already attached to an external executor; skipping internal executor.")
            return

        if MultiThreadedExecutor is None:
            return

        try:
            num_threads = max(os.cpu_count() or 4, 4)
            self._executor = MultiThreadedExecutor(num_threads=num_threads)
            self._executor.add_node(self._node)
            self._executor_thread = threading.Thread(
                target=self._executor.spin,
                daemon=True,
                name="ros2_backend_executor",
            )
            self._executor_thread.start()
            self._log_info(
                f"Background MultiThreadedExecutor started ({num_threads} worker threads) "
                "for concurrent TF and action processing."
            )
        except Exception as exc:
            self._log_warn(f"Could not start background executor: {exc}")
            self._executor = None
            self._executor_thread = None

    def _stop_spinning(self) -> None:
        """Gracefully stop and join the background executor thread."""
        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception as exc:
                self._log_warn(f"Error during executor shutdown: {exc}")

        if self._executor_thread is not None and self._executor_thread.is_alive():
            try:
                self._executor_thread.join(timeout=2.0)
            except Exception as exc:
                self._log_warn(f"Error joining executor thread: {exc}")

        if self._executor is not None and self._node is not None:
            try:
                if hasattr(self._executor, "remove_node"):
                    self._executor.remove_node(self._node)
            except Exception:
                pass

        self._executor_thread = None
        self._executor = None


    def _log_info(self, msg: str) -> None:
        if self._logger is not None and hasattr(self._logger, "info"):
            self._logger.info(f"[Ros2Backend] {msg}")
        else:
            print(f"[Ros2Backend INFO] {msg}")

    def _log_warn(self, msg: str) -> None:
        if self._logger is not None and hasattr(self._logger, "warning"):
            self._logger.warning(f"[Ros2Backend] {msg}")
        elif self._logger is not None and hasattr(self._logger, "warn"):
            self._logger.warn(f"[Ros2Backend] {msg}")
        else:
            print(f"[Ros2Backend WARN] {msg}")

    def _log_error(self, msg: str) -> None:
        if self._logger is not None and hasattr(self._logger, "error"):
            self._logger.error(f"[Ros2Backend] {msg}")
        else:
            print(f"[Ros2Backend ERROR] {msg}")

    def _load_default_database(self) -> Optional[LocationDatabase]:
        """Attempt to load default locations.yaml from package share or config dir."""
        try:
            from ament_index_python.packages import get_package_share_directory
            share_yaml = os.path.join(
                get_package_share_directory("location_navigation"),
                "config",
                "locations.yaml",
            )
            if os.path.isfile(share_yaml):
                return LocationDatabase(share_yaml)
        except Exception:
            pass

        # Check relative directory
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        rel_yaml = os.path.normpath(os.path.join(curr_dir, "..", "config", "locations.yaml"))
        if os.path.isfile(rel_yaml):
            return LocationDatabase(rel_yaml)
        return None

    def _init_action_clients(self) -> None:
        """Initialize Nav2 action clients with ReentrantCallbackGroup to prevent deadlock."""
        if self._node is None or ActionClient is None:
            return

        cb_kwargs = {}
        if self._action_cb_group is not None:
            cb_kwargs["callback_group"] = self._action_cb_group

        if NavigateToPose is not None:
            self._nav_client = ActionClient(self._node, NavigateToPose, self.NAV_ACTION, **cb_kwargs)

        if Spin is not None:
            self._spin_client = ActionClient(self._node, Spin, self.SPIN_ACTION, **cb_kwargs)

        if DriveOnHeading is not None:
            self._drive_client = ActionClient(self._node, DriveOnHeading, self.DRIVE_ACTION, **cb_kwargs)

    def _init_tf(self) -> None:
        """Initialize TF2 buffer and listener for accurate current-pose lookups."""
        if self._node is None or not HAS_TF2 or Buffer is None or TransformListener is None:
            return

        try:
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self._node)
            self._log_info("TF2 Buffer and TransformListener initialized successfully.")
        except Exception as exc:
            self._log_warn(f"Could not initialize TF2 listener: {exc}")

    # ------------------------------------------------------------------
    # Properties & Configuration
    # ------------------------------------------------------------------
    @property
    def fallback_timeout_sec(self) -> float:
        """Configurable timeout in seconds for NavigateToPose fallback execution."""
        return self._fallback_timeout_sec

    @fallback_timeout_sec.setter
    def fallback_timeout_sec(self, val: float) -> None:
        self._fallback_timeout_sec = max(float(val), 1.0)

    # ------------------------------------------------------------------
    # Server Availability Checks
    # ------------------------------------------------------------------
    def wait_for_server(self, action_type: str = "navigate_to_pose", timeout_sec: Optional[float] = None) -> bool:
        timeout = timeout_sec if timeout_sec is not None else self._server_timeout_sec
        client = None

        if action_type == self.NAV_ACTION:
            client = self._nav_client
        elif action_type == self.SPIN_ACTION:
            client = self._spin_client
        elif action_type == self.DRIVE_ACTION:
            client = self._drive_client

        if client is None:
            self._log_warn(f"Action client '{action_type}' is not configured.")
            return False

        self._log_info(f"Checking action server '{action_type}' (timeout={timeout:.1f}s)...")
        ready = client.wait_for_server(timeout_sec=timeout)
        if ready:
            self._log_info(f"Action server '{action_type}' is READY.")
        else:
            self._log_warn(f"Action server '{action_type}' is NOT ready after {timeout:.1f}s.")
        return ready

    # ------------------------------------------------------------------
    # Current Pose Lookup in 'map' Frame (Using TF Time Zero)
    # ------------------------------------------------------------------
    def get_current_robot_pose_in_map(
        self, timeout_sec: float = 5.0
    ) -> Optional[Tuple[float, float, float]]:
        """
        Look up the robot's current pose (x, y, yaw) in the 'map' frame using TF.

        Uses Time() (timestamp 0) to fetch the latest available transform
        without extrapolation errors into the future. Slices the wait so that
        background callbacks can process and cancellation is respected.

        Returns
        -------
        Optional[Tuple[float, float, float]]
            (x, y, yaw_radians) in 'map' frame, or None if lookup fails.
        """
        if self._tf_buffer is None:
            self._log_warn("TF buffer is not initialized; cannot look up current pose in 'map' frame.")
            return None

        start_time = time.time()
        last_exception = None

        while (time.time() - start_time) < timeout_sec:
            if self._cancel_requested:
                self._log_info("Cancellation active; aborting TF pose lookup.")
                return None

            try:
                lookup_time = None
                slice_timeout = None
                if HAS_ROS2 and rclpy is not None:
                    from rclpy.time import Time
                    from rclpy.duration import Duration
                    lookup_time = Time()  # Time 0: latest transform in buffer (no future extrapolation)
                    slice_timeout = Duration(seconds=min(0.5, timeout_sec))

                transform = self._tf_buffer.lookup_transform(
                    target_frame="map",
                    source_frame="base_link",
                    time=lookup_time,
                    timeout=slice_timeout,
                )

                t = transform.transform.translation
                r = transform.transform.rotation

                # Convert quaternion to yaw angle
                siny_cosp = 2.0 * (r.w * r.z + r.x * r.y)
                cosy_cosp = 1.0 - 2.0 * (r.y * r.y + r.z * r.z)
                yaw = math.atan2(siny_cosp, cosy_cosp)

                self._log_info(
                    f"Resolved current robot pose in 'map' frame: x={t.x:.3f}, y={t.y:.3f}, yaw={yaw:.3f} rad ({math.degrees(yaw):.1f}°)"
                )
                return (float(t.x), float(t.y), float(yaw))

            except Exception as exc:
                last_exception = exc
                time.sleep(0.05)

        self._log_warn(f"Transform lookup from 'map' to 'base_link' timed out after {timeout_sec:.1f}s: {last_exception}")
        return None

    def _get_elapsed_time(self, start_wall: float, start_sim: Optional[Any]) -> float:
        """Calculate elapsed time, preferring simulation clock when advancing."""
        if (
            start_sim is not None
            and self._node is not None
            and hasattr(self._node, "get_clock")
        ):
            try:
                start_ns = getattr(start_sim, "nanoseconds", None)
                if isinstance(start_ns, (int, float)) and start_ns > 0:
                    curr_sim = self._node.get_clock().now()
                    curr_ns = getattr(curr_sim, "nanoseconds", None)
                    if (
                        isinstance(curr_ns, (int, float))
                        and curr_ns > 0
                        and curr_sim >= start_sim
                    ):
                        return float((curr_sim - start_sim).nanoseconds) / 1e9
            except Exception:
                pass
        return time.time() - start_wall

    # ------------------------------------------------------------------
    # Future Waiter with Responsive Slicing & Cancel Support
    # ------------------------------------------------------------------
    def _spin_future_responsively(
        self, future: Any, timeout_sec: Optional[float] = None
    ) -> Tuple[bool, bool]:
        start_wall = time.time()
        start_sim = None
        if self._node is not None and hasattr(self._node, "get_clock"):
            try:
                clk = self._node.get_clock().now()
                clk_ns = getattr(clk, "nanoseconds", None)
                if isinstance(clk_ns, (int, float)) and clk_ns > 0:
                    start_sim = clk
            except Exception:
                start_sim = None

        ok_func = rclpy.ok if (rclpy is not None and hasattr(rclpy, "ok")) else (lambda: True)

        while ok_func() and not future.done():
            if self._cancel_requested:
                self._log_info("Cancellation active; interrupting wait.")
                return False, True

            if start_sim is None and self._node is not None and hasattr(self._node, "get_clock"):
                try:
                    clk = self._node.get_clock().now()
                    clk_ns = getattr(clk, "nanoseconds", None)
                    if isinstance(clk_ns, (int, float)) and clk_ns > 0:
                        start_sim = clk
                except Exception:
                    pass

            elapsed = self._get_elapsed_time(start_wall, start_sim)
            if timeout_sec is not None and elapsed > timeout_sec:
                self._log_error(f"Execution timed out after {timeout_sec:.1f} seconds (elapsed: {elapsed:.1f}s).")
                return False, True

            # If node is NOT attached to a background executor, spin until complete if possible
            if (
                self._executor is None
                and rclpy is not None
                and hasattr(rclpy, "spin_until_future_complete")
                and self._node is not None
                and hasattr(self._node, "executor")
                and self._node.executor is None
            ):
                try:
                    rclpy.spin_until_future_complete(self._node, future, timeout_sec=0.1)
                except Exception:
                    time.sleep(0.05)
            else:
                time.sleep(0.05)

        return future.done(), False

    # ------------------------------------------------------------------
    # Quiet Robot Pose Lookup (Map Frame)
    # ------------------------------------------------------------------
    def _lookup_robot_pose_quiet(self) -> Optional[Tuple[float, float, float]]:
        if self._tf_buffer is None:
            return None
        try:
            lookup_time = None
            slice_timeout = None
            if HAS_ROS2 and rclpy is not None:
                from rclpy.time import Time
                from rclpy.duration import Duration
                lookup_time = Time()
                slice_timeout = Duration(seconds=0.1)
            transform = self._tf_buffer.lookup_transform(
                target_frame="map",
                source_frame="base_link",
                time=lookup_time,
                timeout=slice_timeout,
            )
            t = transform.transform.translation
            r = transform.transform.rotation
            siny_cosp = 2.0 * (r.w * r.z + r.x * r.y)
            cosy_cosp = 1.0 - 2.0 * (r.y * r.y + r.z * r.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return (float(t.x), float(t.y), float(yaw))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Progress & Stuck-Aware Future Waiter
    # ------------------------------------------------------------------
    def _spin_future_with_progress(
        self,
        future: Any,
        action_label: str = "Action",
        timeout_sec: Optional[float] = None,
        stuck_timeout_sec: Optional[float] = None,
        max_timeout_sec: Optional[float] = None,
    ) -> Tuple[bool, bool]:
        """
        Wait for action result future while tracking robot progress.

        Allows valid goals that are actively moving towards the target to continue executing,
        while cancelling goals if the robot is stuck (no movement for stuck_timeout_sec)
        or if max_timeout_sec is exceeded.
        """
        if stuck_timeout_sec is None:
            stuck_timeout_sec = self._stuck_timeout_sec
        if max_timeout_sec is None:
            max_timeout_sec = self._max_timeout_sec

        start_wall = time.time()
        start_sim = None
        if self._node is not None and hasattr(self._node, "get_clock"):
            try:
                clk = self._node.get_clock().now()
                clk_ns = getattr(clk, "nanoseconds", None)
                if isinstance(clk_ns, (int, float)) and clk_ns > 0:
                    start_sim = clk
            except Exception:
                start_sim = None

        ok_func = rclpy.ok if (rclpy is not None and hasattr(rclpy, "ok")) else (lambda: True)

        last_progress_wall = start_wall
        last_progress_sim = start_sim
        last_pose = self._lookup_robot_pose_quiet()
        last_check_wall = start_wall
        check_interval = 1.0  # seconds

        while ok_func() and not future.done():
            if self._cancel_requested:
                self._log_info("Cancellation active; interrupting wait.")
                return False, True

            if start_sim is None and self._node is not None and hasattr(self._node, "get_clock"):
                try:
                    clk = self._node.get_clock().now()
                    clk_ns = getattr(clk, "nanoseconds", None)
                    if isinstance(clk_ns, (int, float)) and clk_ns > 0:
                        start_sim = clk
                        last_progress_sim = clk
                except Exception:
                    pass

            elapsed_total = self._get_elapsed_time(start_wall, start_sim)

            # Safety ceiling check
            if max_timeout_sec is not None and elapsed_total > max_timeout_sec:
                self._log_error(
                    f"{action_label} exceeded maximum allowed safety ceiling ({max_timeout_sec:.1f}s). Aborting."
                )
                return False, True

            now_wall = time.time()
            if (now_wall - last_check_wall) >= check_interval:
                last_check_wall = now_wall
                if self._tf_buffer is not None:
                    curr_pose = self._lookup_robot_pose_quiet()
                    if curr_pose is not None and last_pose is not None:
                        dx = curr_pose[0] - last_pose[0]
                        dy = curr_pose[1] - last_pose[1]
                        dist_moved = math.hypot(dx, dy)
                        dyaw = abs(math.atan2(math.sin(curr_pose[2] - last_pose[2]), math.cos(curr_pose[2] - last_pose[2])))
                        if dist_moved >= 0.05 or dyaw >= 0.08:
                            last_progress_wall = now_wall
                            if self._node is not None and hasattr(self._node, "get_clock"):
                                try:
                                    last_progress_sim = self._node.get_clock().now()
                                except Exception:
                                    pass
                            last_pose = curr_pose
                    elif curr_pose is not None and last_pose is None:
                        last_pose = curr_pose
                        last_progress_wall = now_wall

                if last_pose is not None:
                    time_since_progress = self._get_elapsed_time(last_progress_wall, last_progress_sim)
                    grace_period = min(stuck_timeout_sec, timeout_sec if timeout_sec is not None else stuck_timeout_sec)
                    if time_since_progress > stuck_timeout_sec and elapsed_total >= grace_period:
                        self._log_error(
                            f"{action_label} stuck detected: no movement for {time_since_progress:.1f}s "
                            f"(threshold: {stuck_timeout_sec:.1f}s). Cancelling goal."
                        )
                        return False, True
                else:
                    # No TF tracking available: fall back to rigid timeout_sec
                    if timeout_sec is not None and elapsed_total > timeout_sec:
                        self._log_error(
                            f"Execution timed out after {timeout_sec:.1f} seconds (elapsed: {elapsed_total:.1f}s)."
                        )
                        return False, True

            # If node is NOT attached to a background executor, spin until complete if possible
            if (
                self._executor is None
                and rclpy is not None
                and hasattr(rclpy, "spin_until_future_complete")
                and self._node is not None
                and hasattr(self._node, "executor")
                and self._node.executor is None
            ):
                try:
                    rclpy.spin_until_future_complete(self._node, future, timeout_sec=0.1)
                except Exception:
                    time.sleep(0.05)
            else:
                time.sleep(0.05)

        return future.done(), False

    # ------------------------------------------------------------------
    # Action Execution Dispatcher
    # ------------------------------------------------------------------
    def execute_action(self, action: Dict[str, Any]) -> bool:
        if self._node is None:
            self._log_error("ROS 2 node is unavailable.")
            return False

        intent = action.get("intent", "").upper()
        if intent == "ROTATE_LEFT":
            intent = "TURN_LEFT"
        elif intent == "ROTATE_RIGHT":
            intent = "TURN_RIGHT"
        self._cancel_requested = False

        if intent == "MOVE_TO_TARGET":
            return self._execute_move_to_target(action)
        elif intent == "MOVE_FORWARD":
            return self._execute_move_forward(action)
        elif intent == "MOVE_BACKWARD":
            return self._execute_move_backward(action)
        elif intent == "TURN_LEFT":
            return self._execute_turn(action, direction="LEFT")
        elif intent == "TURN_RIGHT":
            return self._execute_turn(action, direction="RIGHT")
        elif intent == "STOP":
            return self.cancel_or_stop()
        elif intent == "MOVE_NEAR_TARGET":
            return self._execute_move_near_target(action)
        elif intent == "MOVE_TOWARD_TARGET":
            return self._reject_unsupported(
                action,
                "MOVE_TOWARD_TARGET is unsupported: approach direction vector and target-relative "
                "distance parameters are undefined in the project schema.",
            )
        else:
            self._log_error(f"Unrecognized action intent '{intent}'.")
            return False

    # ------------------------------------------------------------------
    # MOVE_TO_TARGET Implementation
    # ------------------------------------------------------------------
    def _execute_move_to_target(self, action: Dict[str, Any]) -> bool:
        raw_target = action.get("target", "").strip().lower()
        if not raw_target:
            self._log_error("MOVE_TO_TARGET missing target name.")
            return False

        canonical_target = LEGACY_TARGET_ALIASES.get(raw_target, raw_target)

        if self._db is None:
            self._log_error("Location database is not loaded. Cannot resolve target.")
            return False

        try:
            pose = self._db.get(canonical_target)
        except LocationNotFoundError as exc:
            self._log_error(f"Target '{raw_target}' not found in location database: {exc}")
            return False

        x = pose["x"]
        y = pose["y"]
        yaw = pose.get("yaw", 0.0)
        frame_id = pose.get("frame_id", "map")

        if not self.wait_for_server(self.NAV_ACTION):
            self._log_error(f"Nav2 server '{self.NAV_ACTION}' is unavailable. Aborting navigation.")
            return False

        # Build NavigateToPose goal
        goal_pose = PoseStamped() if PoseStamped is not None else _MockPoseStamped()
        goal_pose.header.frame_id = frame_id
        if hasattr(self._node, "get_clock"):
            goal_pose.header.stamp = self._node.get_clock().now().to_msg()
        goal_pose.pose.position.x = float(x)
        goal_pose.pose.position.y = float(y)
        goal_pose.pose.position.z = 0.0

        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        goal_pose.pose.orientation.x = qx
        goal_pose.pose.orientation.y = qy
        goal_pose.pose.orientation.z = qz
        goal_pose.pose.orientation.w = qw

        if NavigateToPose is not None and hasattr(NavigateToPose, "Goal"):
            goal_msg = NavigateToPose.Goal()
        else:
            goal_msg = _MockGoal()
        goal_msg.pose = goal_pose

        self._log_info(
            f"Sending NavigateToPose goal → {canonical_target} (x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}, frame={frame_id})"
        )

        return self._send_and_await_goal(
            client=self._nav_client,
            goal_msg=goal_msg,
            action_label=f"MOVE_TO_TARGET({canonical_target})",
            timeout_sec=self._fallback_timeout_sec,
        )

    # ------------------------------------------------------------------
    # MOVE_NEAR_TARGET Implementation & Map Verification
    # ------------------------------------------------------------------
    def _is_approach_pose_free_on_map(self, x: float, y: float) -> bool:
        """
        Check static map to ensure the approach pose (x, y) does not collide with an obstacle.
        If map file is not available, returns True (delegating dynamic safety to Nav2 costmaps).
        """
        candidate_paths = [
            os.path.expanduser("~/case_study/room_map.yaml"),
            "/home/reyan/case_study/room_map.yaml",
        ]
        map_yaml = None
        for p in candidate_paths:
            if os.path.isfile(p):
                map_yaml = p
                break

        if map_yaml is None:
            return True

        try:
            origin = None
            resolution = 0.05
            image_rel = None
            with open(map_yaml, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("resolution:"):
                        resolution = float(stripped.split(":", 1)[1].strip())
                    elif stripped.startswith("origin:"):
                        raw = stripped.split(":", 1)[1].strip().strip("[]")
                        origin = [float(v) for v in raw.split(",")]
                    elif stripped.startswith("image:"):
                        image_rel = stripped.split(":", 1)[1].strip()

            if origin is None or image_rel is None:
                return True

            img_path = os.path.join(os.path.dirname(map_yaml), image_rel)
            if not os.path.isfile(img_path):
                return True

            with open(img_path, "rb") as f:
                header = f.readline()
                if not header.startswith(b"P5"):
                    return True
                dims = f.readline().split()
                while dims[0].startswith(b"#"):
                    dims = f.readline().split()
                w, h = int(dims[0]), int(dims[1])
                f.readline()  # 255
                data = f.read()

            gx = int((x - origin[0]) / resolution)
            gy_bot = int((y - origin[1]) / resolution)
            gy_top = h - 1 - gy_bot

            if not (0 <= gx < w and 0 <= gy_top < h):
                self._log_warn(f"Approach pose ({x:.3f}, {y:.3f}) falls outside map boundaries.")
                return False

            pixel_val = data[gy_top * w + gx]
            # Standard ROS 2 map: 254 = free space, 0 = occupied obstacle, 205 = unknown
            if pixel_val < 200:
                self._log_warn(
                    f"Approach pose ({x:.3f}, {y:.3f}) is in occupied or blocked space (pixel={pixel_val})."
                )
                return False

            return True
        except Exception as exc:
            self._log_warn(f"Could not verify approach pose against static map: {exc}")
            return True

    def _execute_move_near_target(self, action: Dict[str, Any]) -> bool:
        raw_target = action.get("target", "").strip().lower()
        if not raw_target:
            self._log_error("MOVE_NEAR_TARGET missing target name.")
            return False

        canonical_target = LEGACY_TARGET_ALIASES.get(raw_target, raw_target)

        if self._db is None:
            self._log_error("Location database is not loaded. Cannot resolve target.")
            return False

        try:
            target_pose = self._db.get(canonical_target)
        except LocationNotFoundError as exc:
            self._log_error(f"Target '{raw_target}' not found in location database: {exc}")
            return False

        # Extract standoff parameter if specified in action
        raw_standoff = action.get("standoff", action.get("standoff_distance", action.get("distance")))
        standoff: Optional[float] = None
        if raw_standoff is not None:
            try:
                standoff = float(raw_standoff)
                if standoff <= 0.0 or not math.isfinite(standoff):
                    self._log_error(f"MOVE_NEAR_TARGET invalid standoff distance {standoff}. Must be positive.")
                    return False
                if standoff > self._max_distance:
                    self._log_error(
                        f"MOVE_NEAR_TARGET standoff {standoff:.2f}m exceeds maximum safety limit {self._max_distance:.2f}m."
                    )
                    return False
            except (ValueError, TypeError):
                self._log_error(f"MOVE_NEAR_TARGET non-numeric standoff parameter: {raw_standoff}")
                return False

        # Check if action explicitly specifies approach_direction
        raw_app_dir = action.get("approach_direction", action.get("approach_yaw"))
        app_dir: Optional[float] = None
        if raw_app_dir is not None:
            try:
                app_dir = float(raw_app_dir)
            except (ValueError, TypeError):
                pass

        # Determine safe approach pose
        approach_pose: Optional[Dict[str, Any]] = None
        x_target = target_pose["x"]
        y_target = target_pose["y"]
        frame_id = target_pose.get("frame_id", "map")

        if app_dir is not None:
            effective_dist = standoff if standoff is not None else target_pose.get("standoff_distance", 0.8)
            ax = x_target + effective_dist * math.cos(app_dir)
            ay = y_target + effective_dist * math.sin(app_dir)
            ayaw = math.atan2(y_target - ay, x_target - ax)
            approach_pose = {"x": float(ax), "y": float(ay), "yaw": float(ayaw), "frame_id": frame_id}
        elif hasattr(self._db, "get_approach_pose"):
            raw_app = self._db.get_approach_pose(canonical_target, standoff=standoff)
            if (
                isinstance(raw_app, dict)
                and "x" in raw_app
                and "y" in raw_app
                and isinstance(raw_app["x"], (int, float))
                and not isinstance(raw_app["x"], bool)
            ):
                approach_pose = raw_app

        if approach_pose is None:
            if "approach_direction" in target_pose and isinstance(target_pose["approach_direction"], (int, float)) and not isinstance(target_pose["approach_direction"], bool):
                angle = float(target_pose["approach_direction"])
                effective_dist = standoff if standoff is not None else target_pose.get("standoff_distance", 0.8)
                ax = x_target + effective_dist * math.cos(angle)
                ay = y_target + effective_dist * math.sin(angle)
                ayaw = target_pose.get("approach_yaw", math.atan2(y_target - ay, x_target - ax))
                approach_pose = {"x": float(ax), "y": float(ay), "yaw": float(ayaw), "frame_id": frame_id}
            elif (
                "approach_x" in target_pose
                and "approach_y" in target_pose
                and isinstance(target_pose["approach_x"], (int, float))
                and not isinstance(target_pose["approach_x"], bool)
                and isinstance(target_pose["approach_y"], (int, float))
                and not isinstance(target_pose["approach_y"], bool)
            ):
                ax = float(target_pose["approach_x"])
                ay = float(target_pose["approach_y"])
                if standoff is not None and standoff > 0.0:
                    dx = ax - x_target
                    dy = ay - y_target
                    cur_d = math.hypot(dx, dy)
                    if cur_d > 1e-4:
                        scale = standoff / cur_d
                        ax = x_target + dx * scale
                        ay = y_target + dy * scale
                ayaw = target_pose.get("approach_yaw", math.atan2(y_target - ay, x_target - ax))
                approach_pose = {"x": float(ax), "y": float(ay), "yaw": float(ayaw), "frame_id": frame_id}

        # Requirement 6: If safe approach pose cannot be determined, return clear error
        if (
            approach_pose is None
            or not isinstance(approach_pose, dict)
            or not isinstance(approach_pose.get("x"), (int, float))
            or isinstance(approach_pose.get("x"), bool)
            or not isinstance(approach_pose.get("y"), (int, float))
            or isinstance(approach_pose.get("y"), bool)
        ):
            self._log_error(
                f"Cannot execute MOVE_NEAR_TARGET for '{raw_target}': safe approach pose cannot be determined. "
                "Neither approach direction nor approach position is configured in the location database, "
                "and no explicit approach vector was specified in action parameters. "
                "Refusing to navigate to exact target coordinates."
            )
            return False

        ax = approach_pose["x"]
        ay = approach_pose["y"]
        ayaw = approach_pose.get("yaw", 0.0)
        aframe = approach_pose.get("frame_id", "map")

        dist_to_target = math.hypot(ax - x_target, ay - y_target)
        if dist_to_target < 0.05:
            self._log_error(
                f"MOVE_NEAR_TARGET calculated approach pose ({ax:.3f}, {ay:.3f}) coincides with target coordinates. "
                "Refusing to navigate directly into target."
            )
            return False

        # Collision avoidance: check static map
        if not self._is_approach_pose_free_on_map(ax, ay):
            self._log_error(
                f"MOVE_NEAR_TARGET calculated approach pose ({ax:.3f}, {ay:.3f}) lies in an occupied obstacle on the map. "
                "Aborting navigation for safety."
            )
            return False

        if not self.wait_for_server(self.NAV_ACTION):
            self._log_error(f"Nav2 server '{self.NAV_ACTION}' is unavailable. Aborting navigation.")
            return False

        # Build NavigateToPose goal
        goal_pose = PoseStamped() if PoseStamped is not None else _MockPoseStamped()
        goal_pose.header.frame_id = aframe
        if hasattr(self._node, "get_clock"):
            goal_pose.header.stamp = self._node.get_clock().now().to_msg()
        goal_pose.pose.position.x = float(ax)
        goal_pose.pose.position.y = float(ay)
        goal_pose.pose.position.z = 0.0

        qx, qy, qz, qw = yaw_to_quaternion(ayaw)
        goal_pose.pose.orientation.x = qx
        goal_pose.pose.orientation.y = qy
        goal_pose.pose.orientation.z = qz
        goal_pose.pose.orientation.w = qw

        if NavigateToPose is not None and hasattr(NavigateToPose, "Goal"):
            goal_msg = NavigateToPose.Goal()
        else:
            goal_msg = _MockGoal()
        goal_msg.pose = goal_pose

        self._log_info(
            f"Sending NavigateToPose goal for MOVE_NEAR_TARGET({canonical_target}) → "
            f"approach pose (x={ax:.3f}, y={ay:.3f}, yaw={ayaw:.3f}, frame={aframe}, standoff={dist_to_target:.2f}m)"
        )

        success = self._send_and_await_goal(
            client=self._nav_client,
            goal_msg=goal_msg,
            action_label=f"MOVE_NEAR_TARGET({canonical_target})",
            timeout_sec=self._fallback_timeout_sec,
        )
        if success:
            return True

        if self._cancel_requested:
            return False

        # If approach was aborted by Nav2 (status code 6 ABORTED due to costmap inflation or obstacle),
        # try adaptive standoff fallback: increase standoff distance to clear obstacles
        aborted_code = 6 if GoalStatus is None else GoalStatus.STATUS_ABORTED
        if self._last_goal_status == aborted_code:
            curr_standoff = dist_to_target
            fallback_standoff = curr_standoff + 0.3
            if fallback_standoff <= self._max_distance:
                self._log_warn(
                    f"MOVE_NEAR_TARGET({canonical_target}) initial approach at {curr_standoff:.2f}m aborted (status=6). "
                    f"Attempting adaptive standoff fallback with increased clearance ({fallback_standoff:.2f}m)..."
                )
                fallback_pose = None
                if hasattr(self._db, "get_approach_pose"):
                    fallback_pose = self._db.get_approach_pose(canonical_target, standoff=fallback_standoff)
                if fallback_pose is not None and "x" in fallback_pose and "y" in fallback_pose:
                    fax = fallback_pose["x"]
                    fay = fallback_pose["y"]
                    fayaw = fallback_pose.get("yaw", 0.0)
                    faframe = fallback_pose.get("frame_id", "map")
                    if self._is_approach_pose_free_on_map(fax, fay):
                        fb_goal = PoseStamped() if PoseStamped is not None else _MockPoseStamped()
                        fb_goal.header.frame_id = faframe
                        if hasattr(self._node, "get_clock"):
                            fb_goal.header.stamp = self._node.get_clock().now().to_msg()
                        fb_goal.pose.position.x = float(fax)
                        fb_goal.pose.position.y = float(fay)
                        fb_goal.pose.position.z = 0.0
                        fb_qx, fb_qy, fb_qz, fb_qw = yaw_to_quaternion(fayaw)
                        fb_goal.pose.orientation.x = fb_qx
                        fb_goal.pose.orientation.y = fb_qy
                        fb_goal.pose.orientation.z = fb_qz
                        fb_goal.pose.orientation.w = fb_qw

                        if NavigateToPose is not None and hasattr(NavigateToPose, "Goal"):
                            fb_goal_msg = NavigateToPose.Goal()
                        else:
                            fb_goal_msg = _MockGoal()
                        fb_goal_msg.pose = fb_goal

                        return self._send_and_await_goal(
                            client=self._nav_client,
                            goal_msg=fb_goal_msg,
                            action_label=f"MOVE_NEAR_TARGET({canonical_target}) [Fallback standoff {fallback_standoff:.2f}m]",
                            timeout_sec=self._fallback_timeout_sec,
                        )
        return False

    # ------------------------------------------------------------------
    # MOVE_FORWARD Implementation
    # ------------------------------------------------------------------
    def _execute_move_forward(self, action: Dict[str, Any]) -> bool:
        dist = action.get("distance")
        if dist is None or not isinstance(dist, (int, float)) or isinstance(dist, bool):
            self._log_error("MOVE_FORWARD missing valid numeric distance parameter.")
            return False

        distance = float(dist)
        if distance <= 0.0 or not math.isfinite(distance):
            self._log_error(f"MOVE_FORWARD distance must be positive and finite, got {distance}.")
            return False

        if distance > self._max_distance:
            self._log_error(
                f"MOVE_FORWARD distance {distance:.2f}m exceeds maximum safety limit {self._max_distance:.2f}m."
            )
            return False

        estimated_duration = (distance / self._linear_speed) + ACTION_TIMEOUT_MARGIN_SEC

        # --------------------------------------------------------------
        # Priority 1: Check Nav2 DriveOnHeading Action Server
        # --------------------------------------------------------------
        if self._drive_client is not None and self.wait_for_server(self.DRIVE_ACTION):
            self._log_info(
                f"Executing MOVE_FORWARD via Nav2 /drive_on_heading: dist={distance:.2f}m, speed={self._linear_speed:.2f}m/s"
            )
            if DriveOnHeading is not None and hasattr(DriveOnHeading, "Goal"):
                goal_msg = DriveOnHeading.Goal()
            else:
                goal_msg = _MockGoal()

            point_cls = Point if Point is not None else _MockPoint
            dur_cls = Duration if Duration is not None else _MockDuration

            goal_msg.target = point_cls(x=distance, y=0.0, z=0.0)
            goal_msg.speed = float(self._linear_speed)
            sec_part = int(estimated_duration)
            nsec_part = int((estimated_duration - sec_part) * 1e9)
            goal_msg.time_allowance = dur_cls(sec=sec_part, nanosec=nsec_part)

            drive_success = self._send_and_await_goal(
                client=self._drive_client,
                goal_msg=goal_msg,
                action_label=f"MOVE_FORWARD({distance:.2f}m via DriveOnHeading)",
                timeout_sec=estimated_duration,
            )
            if drive_success:
                return True

            if self._cancel_requested:
                return False

            self._log_warn(
                "MOVE_FORWARD via /drive_on_heading was not successful. "
                "Attempting fallback to map-frame NavigateToPose."
            )

        # --------------------------------------------------------------
        # Priority 2: Fallback to NavigateToPose using 'map' frame via TF
        # --------------------------------------------------------------
        self._log_info(
            "DriveOnHeading unavailable or failed; attempting fallback to NavigateToPose via 'map' frame current-pose lookup."
        )
        if self._nav_client is not None and self.wait_for_server(self.NAV_ACTION):
            current_pose = self.get_current_robot_pose_in_map(timeout_sec=5.0)
            if current_pose is None:
                self._log_error(
                    "Cannot execute relative MOVE_FORWARD fallback: transform lookup from 'map' to 'base_link' failed. "
                    "Robot pose is unknown. Refusing to guess coordinates."
                )
                return False

            curr_x, curr_y, curr_yaw = current_pose
            # Calculate forward target in 'map' frame along current heading
            target_x = curr_x + distance * math.cos(curr_yaw)
            target_y = curr_y + distance * math.sin(curr_yaw)
            target_yaw = curr_yaw

            self._log_info(
                f"Calculated goal in 'map' frame: from ({curr_x:.3f}, {curr_y:.3f}) -> ({target_x:.3f}, {target_y:.3f}), yaw={target_yaw:.3f} rad"
            )

            goal_pose = PoseStamped() if PoseStamped is not None else _MockPoseStamped()
            goal_pose.header.frame_id = "map"  # Always 'map' frame to match navigation_client.py
            if hasattr(self._node, "get_clock"):
                goal_pose.header.stamp = self._node.get_clock().now().to_msg()
            goal_pose.pose.position.x = float(target_x)
            goal_pose.pose.position.y = float(target_y)
            goal_pose.pose.position.z = 0.0

            qx, qy, qz, qw = yaw_to_quaternion(target_yaw)
            goal_pose.pose.orientation.x = qx
            goal_pose.pose.orientation.y = qy
            goal_pose.pose.orientation.z = qz
            goal_pose.pose.orientation.w = qw

            if NavigateToPose is not None and hasattr(NavigateToPose, "Goal"):
                goal_msg = NavigateToPose.Goal()
            else:
                goal_msg = _MockGoal()
            goal_msg.pose = goal_pose

            # NavigateToPose fallback execution timeout (120s / 2 minutes by default)
            fallback_timeout = max(self._fallback_timeout_sec, estimated_duration)
            self._log_info(
                f"Dispatching NavigateToPose fallback goal for MOVE_FORWARD ({distance:.2f}m) "
                f"with timeout={fallback_timeout:.1f}s."
            )
            return self._send_and_await_goal(
                client=self._nav_client,
                goal_msg=goal_msg,
                action_label=f"MOVE_FORWARD({distance:.2f}m via map-frame NavigateToPose)",
                timeout_sec=fallback_timeout,
            )

        self._log_error(
            f"Neither /{self.DRIVE_ACTION} nor /{self.NAV_ACTION} action server became available "
            f"within server timeout ({self._server_timeout_sec:.1f}s). Aborting MOVE_FORWARD."
        )
        return False

    # ------------------------------------------------------------------
    # MOVE_BACKWARD Implementation
    # ------------------------------------------------------------------
    def _execute_move_backward(self, action: Dict[str, Any]) -> bool:
        dist = action.get("distance", DEFAULT_BACKWARD_DISTANCE)
        if dist is None or not isinstance(dist, (int, float)) or isinstance(dist, bool):
            self._log_error("MOVE_BACKWARD missing valid numeric distance parameter.")
            return False

        distance = float(dist)
        if distance <= 0.0 or not math.isfinite(distance):
            self._log_error(f"MOVE_BACKWARD distance must be positive and finite, got {distance}.")
            return False

        if distance > self._max_distance:
            self._log_error(
                f"MOVE_BACKWARD distance {distance:.2f}m exceeds maximum safety limit {self._max_distance:.2f}m."
            )
            return False

        estimated_duration = (distance / self._linear_speed) + ACTION_TIMEOUT_MARGIN_SEC

        # --------------------------------------------------------------
        # Priority 1: Check Nav2 DriveOnHeading Action Server
        # --------------------------------------------------------------
        if self._drive_client is not None and self.wait_for_server(self.DRIVE_ACTION):
            self._log_info(
                f"Executing MOVE_BACKWARD via Nav2 /drive_on_heading: dist={distance:.2f}m, speed={self._linear_speed:.2f}m/s"
            )
            if DriveOnHeading is not None and hasattr(DriveOnHeading, "Goal"):
                goal_msg = DriveOnHeading.Goal()
            else:
                goal_msg = _MockGoal()

            point_cls = Point if Point is not None else _MockPoint
            dur_cls = Duration if Duration is not None else _MockDuration

            goal_msg.target = point_cls(x=-distance, y=0.0, z=0.0)
            goal_msg.speed = float(self._linear_speed)
            sec_part = int(estimated_duration)
            nsec_part = int((estimated_duration - sec_part) * 1e9)
            goal_msg.time_allowance = dur_cls(sec=sec_part, nanosec=nsec_part)

            drive_success = self._send_and_await_goal(
                client=self._drive_client,
                goal_msg=goal_msg,
                action_label=f"MOVE_BACKWARD({distance:.2f}m via DriveOnHeading)",
                timeout_sec=estimated_duration,
            )
            if drive_success:
                return True

            if self._cancel_requested:
                return False

            self._log_warn(
                "MOVE_BACKWARD via /drive_on_heading was not successful. "
                "Attempting fallback to map-frame NavigateToPose."
            )

        # --------------------------------------------------------------
        # Priority 2: Fallback to NavigateToPose using 'map' frame via TF
        # --------------------------------------------------------------
        self._log_info(
            "DriveOnHeading unavailable or failed; attempting fallback to NavigateToPose via 'map' frame current-pose lookup for MOVE_BACKWARD."
        )
        if self._nav_client is not None and self.wait_for_server(self.NAV_ACTION):
            current_pose = self.get_current_robot_pose_in_map(timeout_sec=5.0)
            if current_pose is None:
                self._log_error(
                    "Cannot execute relative MOVE_BACKWARD fallback: transform lookup from 'map' to 'base_link' failed. "
                    "Robot pose is unknown. Refusing to guess coordinates."
                )
                return False

            curr_x, curr_y, curr_yaw = current_pose
            # Calculate backward target in 'map' frame along opposite of heading
            target_x = curr_x - distance * math.cos(curr_yaw)
            target_y = curr_y - distance * math.sin(curr_yaw)
            target_yaw = curr_yaw

            self._log_info(
                f"Calculated backward goal in 'map' frame: from ({curr_x:.3f}, {curr_y:.3f}) -> ({target_x:.3f}, {target_y:.3f}), yaw={target_yaw:.3f} rad"
            )

            goal_pose = PoseStamped() if PoseStamped is not None else _MockPoseStamped()
            goal_pose.header.frame_id = "map"
            if hasattr(self._node, "get_clock"):
                goal_pose.header.stamp = self._node.get_clock().now().to_msg()
            goal_pose.pose.position.x = float(target_x)
            goal_pose.pose.position.y = float(target_y)
            goal_pose.pose.position.z = 0.0

            qx, qy, qz, qw = yaw_to_quaternion(target_yaw)
            goal_pose.pose.orientation.x = qx
            goal_pose.pose.orientation.y = qy
            goal_pose.pose.orientation.z = qz
            goal_pose.pose.orientation.w = qw

            if NavigateToPose is not None and hasattr(NavigateToPose, "Goal"):
                goal_msg = NavigateToPose.Goal()
            else:
                goal_msg = _MockGoal()
            goal_msg.pose = goal_pose

            fallback_timeout = max(self._fallback_timeout_sec, estimated_duration)
            self._log_info(
                f"Dispatching NavigateToPose fallback goal for MOVE_BACKWARD ({distance:.2f}m) "
                f"with timeout={fallback_timeout:.1f}s."
            )
            return self._send_and_await_goal(
                client=self._nav_client,
                goal_msg=goal_msg,
                action_label=f"MOVE_BACKWARD({distance:.2f}m via map-frame NavigateToPose)",
                timeout_sec=fallback_timeout,
            )

        self._log_error(
            f"Neither /{self.DRIVE_ACTION} nor /{self.NAV_ACTION} action server became available "
            f"within server timeout ({self._server_timeout_sec:.1f}s). Aborting MOVE_BACKWARD."
        )
        return False

    # ------------------------------------------------------------------
    # TURN_LEFT / TURN_RIGHT Implementation
    # ------------------------------------------------------------------
    def _execute_turn(self, action: Dict[str, Any], direction: str) -> bool:
        ang = action.get("angle")
        if ang is None or not isinstance(ang, (int, float)) or isinstance(ang, bool):
            self._log_error(f"TURN_{direction} missing valid numeric angle parameter.")
            return False

        angle_val = float(ang)
        if angle_val <= 0.0 or not math.isfinite(angle_val):
            self._log_error(f"TURN_{direction} angle must be positive and finite, got {angle_val}.")
            return False

        unit = str(action.get("unit", "deg")).strip().lower()
        if unit in {"rad", "radians", "radian"}:
            angle_rad = angle_val
        else:
            angle_rad = math.radians(angle_val)

        if angle_rad > math.radians(MAX_TURN_ANGLE_DEG) + 1e-4:
            self._log_error(f"Turn angle {angle_val} exceeds maximum limit of {MAX_TURN_ANGLE_DEG} degrees.")
            return False

        target_yaw = angle_rad if direction == "LEFT" else -angle_rad
        estimated_duration = (abs(target_yaw) / self._angular_speed) + ACTION_TIMEOUT_MARGIN_SEC

        # --------------------------------------------------------------
        # Priority 1: Check Nav2 Spin Action Server
        # --------------------------------------------------------------
        if self._spin_client is not None and self.wait_for_server(self.SPIN_ACTION):
            self._log_info(
                f"Executing TURN_{direction} via Nav2 /spin: yaw={target_yaw:.3f} rad ({angle_val:.1f}°), speed={self._angular_speed:.2f} rad/s"
            )
            if Spin is not None and hasattr(Spin, "Goal"):
                goal_msg = Spin.Goal()
            else:
                goal_msg = _MockGoal()

            dur_cls = Duration if Duration is not None else _MockDuration
            goal_msg.target_yaw = float(target_yaw)
            sec_part = int(estimated_duration)
            nsec_part = int((estimated_duration - sec_part) * 1e9)
            goal_msg.time_allowance = dur_cls(sec=sec_part, nanosec=nsec_part)

            spin_success = self._send_and_await_goal(
                client=self._spin_client,
                goal_msg=goal_msg,
                action_label=f"TURN_{direction}({angle_val:.1f}° via Spin)",
                timeout_sec=estimated_duration,
            )
            if spin_success:
                return True

            if self._cancel_requested:
                return False

            self._log_warn(
                f"TURN_{direction} via /spin was not successful. "
                "Attempting fallback to in-place rotation via NavigateToPose in 'map' frame."
            )

        # --------------------------------------------------------------
        # Priority 2: Fallback to NavigateToPose using 'map' frame via TF
        # --------------------------------------------------------------
        self._log_info(
            f"Spin unavailable or failed; attempting fallback to NavigateToPose in 'map' frame for TURN_{direction}."
        )
        if self._nav_client is not None and self.wait_for_server(self.NAV_ACTION):
            current_pose = self.get_current_robot_pose_in_map(timeout_sec=5.0)
            if current_pose is None:
                self._log_error(
                    f"Cannot execute relative TURN_{direction} fallback: transform lookup from 'map' to 'base_link' failed. "
                    "Robot pose is unknown. Refusing to guess coordinates."
                )
                return False

            curr_x, curr_y, curr_yaw = current_pose
            new_yaw = normalize_yaw(curr_yaw + target_yaw)

            self._log_info(
                f"Calculated in-place turn in 'map' frame: at ({curr_x:.3f}, {curr_y:.3f}), new_yaw={new_yaw:.3f} rad ({math.degrees(new_yaw):.1f}°)"
            )

            goal_pose = PoseStamped() if PoseStamped is not None else _MockPoseStamped()
            goal_pose.header.frame_id = "map"
            if hasattr(self._node, "get_clock"):
                goal_pose.header.stamp = self._node.get_clock().now().to_msg()
            goal_pose.pose.position.x = float(curr_x)
            goal_pose.pose.position.y = float(curr_y)
            goal_pose.pose.position.z = 0.0

            qx, qy, qz, qw = yaw_to_quaternion(new_yaw)
            goal_pose.pose.orientation.x = qx
            goal_pose.pose.orientation.y = qy
            goal_pose.pose.orientation.z = qz
            goal_pose.pose.orientation.w = qw

            if NavigateToPose is not None and hasattr(NavigateToPose, "Goal"):
                goal_msg = NavigateToPose.Goal()
            else:
                goal_msg = _MockGoal()
            goal_msg.pose = goal_pose

            # NavigateToPose fallback execution timeout (120s / 2 minutes by default)
            fallback_timeout = max(self._fallback_timeout_sec, estimated_duration)
            self._log_info(
                f"Dispatching NavigateToPose fallback goal for TURN_{direction} ({angle_val:.1f}°) "
                f"with timeout={fallback_timeout:.1f}s."
            )
            return self._send_and_await_goal(
                client=self._nav_client,
                goal_msg=goal_msg,
                action_label=f"TURN_{direction}({angle_val:.1f}° via map-frame NavigateToPose)",
                timeout_sec=fallback_timeout,
            )

        self._log_error(
            f"Neither /{self.SPIN_ACTION} nor /{self.NAV_ACTION} action server became available "
            f"within server timeout ({self._server_timeout_sec:.1f}s). Aborting TURN_{direction}."
        )
        return False

    # ------------------------------------------------------------------
    # Unsupported Action Rejection
    # ------------------------------------------------------------------
    def _reject_unsupported(self, action: Dict[str, Any], reason: str) -> bool:
        self._log_error(f"UNSUPPORTED ACTION '{action.get('intent')}': {reason}")
        return False

    # ------------------------------------------------------------------
    # Generic Action Goal Dispatch & Monitoring
    # ------------------------------------------------------------------
    def _send_and_await_goal(
        self, client: Any, goal_msg: Any, action_label: str, timeout_sec: float
    ) -> bool:
        self._active_goal_handle = None
        self._last_goal_status = None

        send_future = client.send_goal_async(goal_msg)
        completed, cancelled_or_timed_out = self._spin_future_responsively(
            send_future, timeout_sec=self._server_timeout_sec
        )

        if not completed or cancelled_or_timed_out:
            self._log_error(f"Failed to receive goal response for {action_label}.")
            return False

        goal_handle = send_future.result()
        if goal_handle is None or not getattr(goal_handle, "accepted", False):
            self._log_error(f"Goal for {action_label} was REJECTED by action server.")
            return False

        self._log_info(f"Goal for {action_label} ACCEPTED. Executing...")
        self._active_goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        completed, cancelled_or_timed_out = self._spin_future_with_progress(
            result_future,
            action_label=action_label,
            timeout_sec=timeout_sec,
            stuck_timeout_sec=self._stuck_timeout_sec,
            max_timeout_sec=self._max_timeout_sec,
        )

        if cancelled_or_timed_out:
            self._log_warn(f"{action_label} was interrupted, cancelled, or timed out.")
            self._cancel_active_goal(goal_handle)
            self._active_goal_handle = None
            return False

        self._active_goal_handle = None
        result_response = result_future.result()
        status = result_response.status
        self._last_goal_status = status

        succeeded_code = 4 if GoalStatus is None else GoalStatus.STATUS_SUCCEEDED

        if status == succeeded_code:
            self._log_info(f"✓ {action_label} SUCCEEDED.")
            return True
        else:
            self._log_error(f"✗ {action_label} FAILED (status code: {status}).")
            return False

    # ------------------------------------------------------------------
    # Cancellation & STOP Handling
    # ------------------------------------------------------------------
    def cancel_or_stop(self) -> bool:
        self._cancel_requested = True
        self._log_info("Software STOP / cancel requested.")

        if self._active_goal_handle is not None:
            self._cancel_active_goal(self._active_goal_handle)
            self._active_goal_handle = None
            return True

        self._log_info("No active goal handle to cancel.")
        return True

    def _cancel_active_goal(self, goal_handle: Any) -> bool:
        try:
            self._log_info("Dispatching cancellation to active Nav2 goal...")
            cancel_future = goal_handle.cancel_goal_async()
            completed, _ = self._spin_future_responsively(cancel_future, timeout_sec=5.0)
            if completed:
                self._log_info("Goal cancellation request completed.")
                return True
            else:
                self._log_warn("Goal cancellation timed out waiting for server response.")
                return False
        except Exception as exc:
            self._log_error(f"Exception while cancelling goal: {exc}")
            return False

    def destroy(self) -> None:
        """Cleanly shut down action clients, executor, background thread, and node."""
        if self._active_goal_handle is not None:
            try:
                self.cancel_or_stop()
            except Exception:
                pass

        self._stop_spinning()

        for client in (self._nav_client, self._spin_client, self._drive_client):
            if client is not None and hasattr(client, "destroy"):
                try:
                    client.destroy()
                except Exception:
                    pass

        if self._owns_node and self._node is not None:
            if hasattr(self._node, "destroy_node"):
                try:
                    self._node.destroy_node()
                except Exception:
                    pass
            self._node = None

