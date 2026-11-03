#!/usr/bin/env python3
"""
# Extracts the navigation plan JSON from command-line arguments.
# Validates the JSON input and checks its structure.
# Loads target location information from locations.yaml.
# Routes sequential plans and motion commands to SequentialActionExecutor.
# Resolves the target pose and sends single-target navigation to NavigationClient.
# Aborts navigation for unknown targets or unsupported intents.
# Prints the execution success or failure result.
# Destroys ROS nodes and shuts down ROS 2.

"""
# Imports ROS 2 libraries

import os
import sys
import json
from typing import Optional

try:
    import rclpy
    from rclpy.node import Node
except ImportError:
    rclpy = None
    Node = None

# Finds the installed ROS package directory.

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    get_package_share_directory = None


from location_navigation.location_database import LocationDatabase, LocationNotFoundError
from location_navigation.navigation_client import NavigationClient
from location_navigation.sequential_executor import SequentialActionExecutor
from location_navigation.ros2_backend import (
    Ros2ActionExecutionBackend,
    LEGACY_TARGET_ALIASES,
    DEFAULT_FALLBACK_TIMEOUT_SEC,
    detect_use_sim_time,
)


def create_nav_bridge_node(node_name: str = "llm_sequential_nav_bridge") -> Optional[Node]:
    """Create bridge node with simulation-time awareness."""
    if Node is None or rclpy is None:
        return None

    use_sim = detect_use_sim_time()
    node = None
    try:
        from rclpy.parameter import Parameter
        if use_sim:
            node = Node(node_name, parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        else:
            node = Node(node_name)
    except Exception:
        try:
            node = Node(node_name)
        except Exception:
            node = None

    # If simulation time was not set earlier, check whether the new ROS node needs to use it.
    if not use_sim and node is not None:
        if detect_use_sim_time(node):
            try:
                from rclpy.parameter import Parameter
                node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
            except Exception:
                pass

    return node


def get_fallback_timeout_sec(node: Optional[Node] = None) -> float:
    """    # Gets the timeout from environment variables.
    # If unavailable, checks the ROS parameter.
    # Uses the default timeout if no value is set.
    # Ensures the timeout is at least 1 second.
    """
    # 1. Environment variable override
    env_val = os.environ.get("NAV_FALLBACK_TIMEOUT_SEC") or os.environ.get("FALLBACK_TIMEOUT_SEC")
    if env_val:
        try:
            return max(float(env_val), 1.0)
        except ValueError:
            pass

    # 2. Node parameter override
    if node is not None and hasattr(node, "get_parameter"):
        try:
            if hasattr(node, "has_parameter") and node.has_parameter("fallback_timeout_sec") is True:
                param = node.get_parameter("fallback_timeout_sec")
                if param is not None and hasattr(param, "value"):
                    val = param.value
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        return max(float(val), 1.0)
                    elif isinstance(val, str):
                        try:
                            return max(float(val), 1.0)
                        except ValueError:
                            pass
            elif hasattr(node, "declare_parameter"):
                node.declare_parameter("fallback_timeout_sec", DEFAULT_FALLBACK_TIMEOUT_SEC)
        except Exception:
            pass

    return DEFAULT_FALLBACK_TIMEOUT_SEC


def resolve_locations_yaml_path() -> str:
    """Find absolute path to locations.yaml in package share or workspace."""
    if get_package_share_directory is not None:
        try:
            pkg_share = get_package_share_directory('location_navigation')
            share_yaml = os.path.join(pkg_share, 'config', 'locations.yaml')
            if os.path.isfile(share_yaml):
                return share_yaml
        except Exception:
            pass

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    rel_yaml = os.path.normpath(os.path.join(curr_dir, '..', 'config', 'locations.yaml'))
    if os.path.isfile(rel_yaml):
        return rel_yaml

    for p in [
        os.path.expanduser('~/case_study/src/location_navigation/config/locations.yaml'),
        '/home/reyan/case_study/src/location_navigation/config/locations.yaml',
    ]:
        if os.path.isfile(p):
            return p

    return '/home/reyan/case_study/src/location_navigation/config/locations.yaml'


def extract_plan_argument(argv: list) -> Optional[str]:
    """Extract JSON string argument from argv, ignoring ROS 2 arguments."""
    non_ros_args = []
    in_ros_args = False
    for arg in argv[1:]:
        if arg == "--ros-args":
            in_ros_args = True
            continue
        if in_ros_args:
            if arg == "--":
                in_ros_args = False
                continue
            if arg.startswith("-") or ":=" in arg:
                continue
        non_ros_args.append(arg)

    for arg in non_ros_args:
        t = arg.strip()
        if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
            return arg

    return non_ros_args[0] if non_ros_args else None


def main(args=None):
    if rclpy is None:
        print("[llm_to_nav] Error: rclpy is not available. Please source your ROS 2 Jazzy environment.")
        sys.exit(1)

    rclpy.init(args=args)

    # 1. Parse CLI arguments
    json_str = extract_plan_argument(sys.argv)
    if json_str is None:
        print("Usage: ros2 run location_navigation llm_to_nav '<json_string>'")
        print("Examples:")
        print("  ros2 run location_navigation llm_to_nav '{\"intent\":\"MOVE_TO_TARGET\",\"target\":\"table\"}'")
        print("  ros2 run location_navigation llm_to_nav '{\"intent\":\"SEQUENTIAL\",\"step_count\":2,\"action_sequence\":[{\"intent\":\"MOVE_FORWARD\",\"distance\":1.0},{\"intent\":\"MOVE_TO_TARGET\",\"target\":\"table\"}]}'")
        rclpy.shutdown()
        return

    # 2. Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[llm_to_nav] Error parsing JSON: {e}")
        rclpy.shutdown()
        sys.exit(1)

    if not isinstance(data, dict):
        print(f"[llm_to_nav] Expected JSON object, got {type(data).__name__}.")
        rclpy.shutdown()
        sys.exit(1)

    intent = data.get("intent", "").strip().upper()
    yaml_path = resolve_locations_yaml_path()

    try:
        db = LocationDatabase(yaml_path)
    except Exception as exc:
        print(f"[llm_to_nav] Failed to load location database from {yaml_path}: {exc}")
        rclpy.shutdown()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Case A: Multi-step Sequential Plan or Single Motion Action
    # -----------------------------------------------------------------------
    is_sequential = (intent == "SEQUENTIAL") or ("action_sequence" in data)
    is_motion_or_special = intent in {"MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"}
    is_approach_near = intent in {"MOVE_NEAR_TARGET", "MOVE_TOWARD_TARGET"}

    if is_sequential or is_motion_or_special or is_approach_near:
        print(f"[llm_to_nav] Routing request through SequentialActionExecutor (intent: {intent})...")
        node = create_nav_bridge_node("llm_sequential_nav_bridge")
        node_logger = node.get_logger() if node is not None and hasattr(node, "get_logger") else None

        fallback_timeout = get_fallback_timeout_sec(node)
        print(f"[llm_to_nav] Fallback NavigateToPose execution timeout configured: {fallback_timeout:.1f}s")

        backend = Ros2ActionExecutionBackend(
            node=node,
            location_database=db,
            locations_yaml_path=yaml_path,
            fallback_timeout_sec=fallback_timeout,
            logger=node_logger,
        )

        executor = SequentialActionExecutor(
            known_locations=db.list_locations(),
            backend=backend,
            logger=node_logger,
        )

        exec_res = executor.execute_plan(data)

        if exec_res.success:
            print(f"[llm_to_nav] ✓ Sequential execution succeeded: {exec_res.completed_steps}/{exec_res.total_steps} steps completed.")
        else:
            print(f"[llm_to_nav] ✗ Execution failed at step {exec_res.failed_step_index}: {exec_res.message}")

        backend.destroy()
        if node is not None and hasattr(node, "destroy_node"):
            try:
                node.destroy_node()
            except Exception:
                pass
        rclpy.shutdown()
        sys.exit(0 if exec_res.success else 1)

    # -----------------------------------------------------------------------
    # Case B: Single-Target Navigation (MOVE_TO_TARGET)
    # -----------------------------------------------------------------------
    if intent == "MOVE_TO_TARGET":
        target = data.get("target", "").strip().lower()
        if not target:
            print("[llm_to_nav] No 'target' specified in JSON.")
            rclpy.shutdown()
            sys.exit(1)

        # Resolve legacy aliases (kitchen -> refrigerator, bedroom -> bed, etc.)
        canonical_target = LEGACY_TARGET_ALIASES.get(target, target)
        print(f"[llm_to_nav] Parsed single-target request to move to: '{target}' (resolved: '{canonical_target}')")

        try:
            pose = db.get(canonical_target)
        except LocationNotFoundError as exc:
            print(f"[llm_to_nav] Location error: {exc}")
            rclpy.shutdown()
            sys.exit(1)

        use_sim = detect_use_sim_time()
        nav_client = NavigationClient(use_sim_time=use_sim)

        if not nav_client.wait_for_server(timeout_sec=15.0):
            nav_client.destroy_node()
            rclpy.shutdown()
            sys.exit(1)

        fallback_timeout = get_fallback_timeout_sec(nav_client)
        success = nav_client.navigate_to_pose(
            x=pose['x'],
            y=pose['y'],
            yaw=pose.get('yaw', 0.0),
            frame_id=pose.get('frame_id', 'map'),
            location_name=canonical_target,
            timeout_sec=fallback_timeout,
        )

        if success:
            print(f"[llm_to_nav] Successfully reached {target}!")
        else:
            print(f"[llm_to_nav] Failed to reach {target}.")

        nav_client.destroy_node()
        rclpy.shutdown()
        sys.exit(0 if success else 1)

    # -----------------------------------------------------------------------
    # Case C: Safe Abort on Unknown Target or Invalid Intent
    # -----------------------------------------------------------------------
    if intent == "UNKNOWN_TARGET":
        print("[llm_to_nav] Safe error: UNKNOWN_TARGET. Aborting navigation.")
        rclpy.shutdown()
        sys.exit(1)

    print(f"[llm_to_nav] Unsupported or unrecognized intent '{intent}'. Expected 'SEQUENTIAL' or 'MOVE_TO_TARGET'.")
    rclpy.shutdown()
    sys.exit(1)


if __name__ == '__main__':
    main()
