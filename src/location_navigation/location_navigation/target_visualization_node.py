#!/usr/bin/env python3
"""
Target Visualization Node for car_bot Autonomous Robot Project.

Responsibilities:
- Reads target locations and coordinates from locations.yaml via LocationDatabase.
- Publishes ROS 2 MarkerArray on /target_markers for visualization in RViz2.
- Displays target pin markers with distinct colors, 3D text labels, and approach poses.
- Subscribes to /selected_target to dynamically highlight user-requested targets.
- Subscribes to /plan (Nav2 Path) and /amcl_pose (Robot pose) to visualize active navigation paths.
- Automatically generates high-resolution PNG map visualizations on startup and when targets are selected.
"""

import os
import sys
import math
from typing import Dict, List, Optional, Tuple

# Ensure custom Python libraries (Pillow/PyYAML) can be imported
for extra_path in [
    os.path.expanduser('~/snap/antigravity-cli/common/local/lib/python3.14/dist-packages'),
    os.path.expanduser('~/.local/lib/python3.12/site-packages'),
]:
    if os.path.isdir(extra_path) and extra_path not in sys.path:
        sys.path.insert(0, extra_path)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point, PoseWithCovarianceStamped
    from std_msgs.msg import String, ColorRGBA
    from nav_msgs.msg import Path
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object

from location_navigation.location_database import LocationDatabase, TARGET_ALIASES
from location_navigation.map_png_generator import generate_map_visualization


# Target color definitions for RViz2 (RGBA normalized to 0.0 - 1.0)
RVIZ_TARGET_COLORS = {
    "refrigerator": {"r": 0.18, "g": 0.80, "b": 0.44, "label": "Kitchen / Refrigerator"},
    "kitchen":      {"r": 0.18, "g": 0.80, "b": 0.44, "label": "Kitchen"},
    "bed":          {"r": 0.20, "g": 0.60, "b": 0.86, "label": "Bedroom / Bed"},
    "bedroom":      {"r": 0.20, "g": 0.60, "b": 0.86, "label": "Bedroom"},
    "sofa":         {"r": 0.90, "g": 0.49, "b": 0.13, "label": "Sofa / Living Area"},
    "tv":           {"r": 0.61, "g": 0.35, "b": 0.71, "label": "TV Area"},
    "table":        {"r": 0.91, "g": 0.30, "b": 0.24, "label": "Dining Table"},
    "dining_table": {"r": 0.95, "g": 0.61, "b": 0.07, "label": "Dining Area"},
}

DEFAULT_RVIZ_COLOR = {"r": 0.4, "g": 0.5, "b": 0.6, "label": "Target"}


class TargetVisualizationNode(Node):
    """
    ROS 2 Node that publishes target markers to RViz2 and manages map PNG export.
    """

    def __init__(self):
        super().__init__('target_visualization_node')

        # Declare parameters
        self.declare_parameter('locations_yaml', '/home/reyan/case_study/src/location_navigation/config/locations.yaml')
        self.declare_parameter('map_yaml', '/home/reyan/case_study/room_map.yaml')
        self.declare_parameter('output_dir', '/home/reyan/case_study/output')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('publish_rate_hz', 1.0)
        self.declare_parameter('auto_generate_png', True)

        self.locations_yaml = self.get_parameter('locations_yaml').get_parameter_value().string_value
        self.map_yaml = self.get_parameter('map_yaml').get_parameter_value().string_value
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        publish_rate = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        self.auto_generate_png = self.get_parameter('auto_generate_png').get_parameter_value().bool_value

        self.selected_target: Optional[str] = None
        self.robot_current_pose: Optional[Tuple[float, float, float]] = None
        self.latest_plan_points: Optional[List[Tuple[float, float]]] = None

        # Load Location Database
        try:
            self.db = LocationDatabase(self.locations_yaml)
            self.get_logger().info(f'Loaded {len(self.db.list_locations())} targets from: {self.locations_yaml}')
        except Exception as e:
            self.get_logger().error(f'Failed to load LocationDatabase: {e}')
            self.db = None

        # Latched QoS Profile for markers so late-joining RViz instances see them immediately
        marker_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=10
        )

        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, '/target_markers', marker_qos)
        self.selected_info_pub = self.create_publisher(String, '/selected_target_info', 10)

        # Subscribers
        self.create_subscription(String, '/selected_target', self.selected_target_callback, 10)
        self.create_subscription(Path, '/plan', self.plan_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_pose_callback, 10)

        # Periodic timer for marker publication
        timer_period = 1.0 / max(publish_rate, 0.1)
        self.timer = self.create_timer(timer_period, self.publish_markers)

        # Initial Marker & PNG generation
        self.publish_markers()
        if self.auto_generate_png:
            self.generate_png_visualization()

        self.get_logger().info('TargetVisualizationNode initialized successfully.')

    def selected_target_callback(self, msg: String):
        """Called when a user or agent selects a target to highlight."""
        req_name = msg.data.strip()
        if not req_name:
            self.selected_target = None
            self.get_logger().info('Cleared selected target.')
        else:
            self.selected_target = req_name
            self.get_logger().info(f'Selected target updated to: "{self.selected_target}"')

        # Re-publish markers immediately
        self.publish_markers()

        # Update PNG export
        if self.auto_generate_png and self.selected_target:
            self.generate_png_visualization(selected=self.selected_target)

    def plan_callback(self, msg: Path):
        """Captures Nav2 planned path for visualization overlay."""
        if not msg.poses:
            return
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.latest_plan_points = pts
        self.get_logger().info(f'Received active Nav2 plan with {len(pts)} waypoints.')

        if self.auto_generate_png:
            self.generate_png_visualization(selected=self.selected_target)

    def amcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        """Tracks robot's current pose from AMCL."""
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        yaw = 2.0 * math.atan2(qz, qw)
        self.robot_current_pose = (px, py, yaw)

    def generate_png_visualization(self, selected: Optional[str] = None):
        """Generates high-resolution PNG maps."""
        try:
            out_file = os.path.join(self.output_dir, "target_map_visualization.png")
            if selected:
                clean_name = selected.strip().lower().replace(" ", "_")
                out_file = os.path.join(self.output_dir, f"selected_target_{clean_name}.png")

            generate_map_visualization(
                map_yaml_path=self.map_yaml,
                locations_yaml_path=self.locations_yaml,
                output_png_path=out_file,
                selected_target=selected or self.selected_target,
                robot_start_pose=(0.0, 0.0, 0.0),
                robot_current_pose=self.robot_current_pose,
                planned_path=self.latest_plan_points,
            )
        except Exception as e:
            self.get_logger().warn(f'Could not generate PNG map visualization: {e}')

    def publish_markers(self):
        """Constructs and publishes MarkerArray with pins, labels, and approach poses."""
        if self.db is None:
            return

        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        locations = self.db.list_locations()

        # Resolve selected target canonical key
        canonical_sel = None
        if self.selected_target:
            raw_sel = self.selected_target.strip().lower()
            canonical_sel = TARGET_ALIASES.get(raw_sel, raw_sel)

        marker_id = 0

        for name in locations:
            try:
                pose = self.db.get(name)
            except Exception:
                continue

            x = pose['x']
            y = pose['y']
            frame = pose.get('frame_id', self.map_frame)
            is_selected = (name == canonical_sel) or (self.selected_target and self.selected_target.lower() == name)

            color_cfg = RVIZ_TARGET_COLORS.get(name, DEFAULT_RVIZ_COLOR)
            display_name = color_cfg.get('label', name.capitalize())

            # -------------------------------------------------------------
            # 1. Target Pin (Cylinder Marker)
            # -------------------------------------------------------------
            pin = Marker()
            pin.header.frame_id = frame
            pin.header.stamp = now
            pin.ns = "target_pins"
            pin.id = marker_id
            marker_id += 1
            pin.type = Marker.CYLINDER
            pin.action = Marker.ADD

            pin.pose.position.x = x
            pin.pose.position.y = y
            pin.pose.position.z = 0.20 if is_selected else 0.12
            pin.pose.orientation.w = 1.0

            pin.scale.x = 0.45 if is_selected else 0.30
            pin.scale.y = 0.45 if is_selected else 0.30
            pin.scale.z = 0.40 if is_selected else 0.24

            if is_selected:
                # Highlight in bright gold
                pin.color.r = 1.0
                pin.color.g = 0.84
                pin.color.b = 0.0
                pin.color.a = 0.95
            else:
                pin.color.r = float(color_cfg['r'])
                pin.color.g = float(color_cfg['g'])
                pin.color.b = float(color_cfg['b'])
                pin.color.a = 0.85

            marker_array.markers.append(pin)

            # -------------------------------------------------------------
            # 2. Text Label (TEXT_VIEW_FACING Marker)
            # -------------------------------------------------------------
            text_marker = Marker()
            text_marker.header.frame_id = frame
            text_marker.header.stamp = now
            text_marker.ns = "target_labels"
            text_marker.id = marker_id
            marker_id += 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = x
            text_marker.pose.position.y = y
            text_marker.pose.position.z = 0.55 if is_selected else 0.40
            text_marker.pose.orientation.w = 1.0

            text_marker.scale.z = 0.28 if is_selected else 0.22

            if is_selected:
                text_marker.text = f"★ {display_name} [{x:.2f}, {y:.2f}]"
                text_marker.color.r = 1.0
                text_marker.color.g = 0.95
                text_marker.color.b = 0.2
                text_marker.color.a = 1.0
            else:
                text_marker.text = f"{display_name}"
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = 0.95

            marker_array.markers.append(text_marker)

            # -------------------------------------------------------------
            # 3. Approach Pose Marker (Arrow pointing towards target)
            # -------------------------------------------------------------
            if 'approach_x' in pose and 'approach_y' in pose:
                ax = pose['approach_x']
                ay = pose['approach_y']

                arrow = Marker()
                arrow.header.frame_id = frame
                arrow.header.stamp = now
                arrow.ns = "approach_poses"
                arrow.id = marker_id
                marker_id += 1
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD

                p_start = Point(x=ax, y=ay, z=0.05)
                p_end = Point(x=x, y=y, z=0.05)
                arrow.points = [p_start, p_end]

                arrow.scale.x = 0.05  # shaft diameter
                arrow.scale.y = 0.12  # head diameter
                arrow.scale.z = 0.15  # head length

                arrow.color.r = 0.8
                arrow.color.g = 0.8
                arrow.color.b = 0.8
                arrow.color.a = 0.70

                marker_array.markers.append(arrow)

        self.marker_pub.publish(marker_array)


def main(args=None):
    if not HAS_ROS2:
        print("[target_visualization_node] Error: rclpy not found. Please source ROS 2 Jazzy.")
        sys.exit(1)

    rclpy.init(args=args)
    node = TargetVisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
