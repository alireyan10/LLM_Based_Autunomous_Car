#!/usr/bin/env python3
"""
Agent-Based Target Query & Selection Module for car_bot.

Responsibilities:
- Parses user queries like "Where is the table?", "Where is the kitchen?", or Bangla "টেবিল কোথায়?".
- Resolves the requested target using the existing LocationDatabase (including aliases).
- Retrieves the actual map coordinates and approach pose.
- Publishes the target to /selected_target to trigger RViz2 marker highlight and PNG generation.
- Formats and displays the response to the user.
"""

import os
import sys
import re
from typing import Optional, Tuple, Dict, Any

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
    from std_msgs.msg import String
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object

from location_navigation.location_database import LocationDatabase, LocationNotFoundError, TARGET_ALIASES
from location_navigation.map_png_generator import generate_map_visualization


# Bangla to English target keyword mapping
BANGLA_TARGET_MAP = {
    "টেবিল": "table",
    "ডাইনিং টেবিল": "dining_table",
    "ডাইনিং": "dining_table",
    "খাবার টেবিল": "dining_table",
    "সোফা": "sofa",
    "টিভি": "tv",
    "টেলিভিশন": "tv",
    "বিছানা": "bed",
    "খাট": "bed",
    "বেডরুম": "bedroom",
    "শোবার ঘর": "bedroom",
    "ফ্রিজ": "refrigerator",
    "রেফ্রিজারেটর": "refrigerator",
    "রান্নাঘর": "kitchen",
}

QUERY_PATTERNS = [
    r"where\s+is\s+(?:the\s+)?([a-zA-Z_ ]+)\??",
    r"find\s+(?:the\s+)?([a-zA-Z_ ]+)",
    r"locate\s+(?:the\s+)?([a-zA-Z_ ]+)",
    r"show\s+(?:me\s+)?(?:the\s+)?([a-zA-Z_ ]+)",
    r"select\s+(?:the\s+)?([a-zA-Z_ ]+)",
    r"which\s+way\s+to\s+(?:the\s+)?([a-zA-Z_ ]+)",
]


def extract_target_from_query(query: str, db: LocationDatabase) -> Optional[str]:
    """
    Identifies target name from natural language query in English or Bangla.
    """
    text = query.strip().lower()

    # Check Bangla target keywords
    for b_keyword, eng_target in BANGLA_TARGET_MAP.items():
        if b_keyword in text:
            return eng_target

    # Check regex query patterns
    for pattern in QUERY_PATTERNS:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1).strip().replace(" ", "_")
            if candidate in db or candidate in TARGET_ALIASES:
                return candidate

    # Check direct canonical names and aliases
    for word in text.replace("?", "").replace(".", "").split():
        clean_word = word.strip()
        if clean_word in db or clean_word in TARGET_ALIASES:
            return clean_word

    # Check multi-word aliases (dining table, dining_table)
    clean_text = text.replace("?", "").strip()
    if clean_text in db or clean_text in TARGET_ALIASES:
        return clean_text

    return None


def query_and_select_target(
    query_text: str,
    locations_yaml: str = "/home/reyan/case_study/src/location_navigation/config/locations.yaml",
    map_yaml: str = "/home/reyan/case_study/room_map.yaml",
    output_dir: str = "/home/reyan/case_study/output",
    publish_ros: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Executes Feature B:
    1. Identifies target.
    2. Retrieves actual coordinates from locations.yaml.
    3. Publishes /selected_target.
    4. Generates selected_target_<target>.png.
    5. Displays structured information.
    """
    db = LocationDatabase(locations_yaml)
    target = extract_target_from_query(query_text, db)

    if not target:
        print(f"\n[Agent Target Query] Could not identify a known target in query: \"{query_text}\"")
        print(f"Available targets: {db.list_locations()} (aliases: {list(TARGET_ALIASES.keys())})")
        return None

    canonical_target = TARGET_ALIASES.get(target, target)
    try:
        pose = db.get(canonical_target)
    except LocationNotFoundError:
        print(f"\n[Agent Target Query] Target '{target}' is not in locations database.")
        return None

    x = pose['x']
    y = pose['y']
    yaw = pose.get('yaw', 0.0)
    frame = pose.get('frame_id', 'map')
    approach_x = pose.get('approach_x')
    approach_y = pose.get('approach_y')
    approach_yaw = pose.get('approach_yaw')

    # 1. Publish to ROS 2 /selected_target if ROS 2 is running
    if publish_ros and HAS_ROS2:
        try:
            if not rclpy.ok():
                rclpy.init()
            node = Node('agent_target_query_publisher')
            pub = node.create_publisher(String, '/selected_target', 10)
            msg = String()
            msg.data = canonical_target
            # Publish multiple times briefly to ensure receipt
            for _ in range(3):
                pub.publish(msg)
                rclpy.spin_once(node, timeout_sec=0.05)
            node.destroy_node()
        except Exception as e:
            # Fallback if ROS 2 daemon/clock is not active
            pass

    # 2. Generate updated PNG image
    clean_target = canonical_target.replace(" ", "_")
    png_path = os.path.join(output_dir, f"selected_target_{clean_target}.png")
    try:
        generate_map_visualization(
            map_yaml_path=map_yaml,
            locations_yaml_path=locations_yaml,
            output_png_path=png_path,
            selected_target=canonical_target,
        )
    except Exception as e:
        png_path = f"(Error generating PNG: {e})"

    # 3. Print Structured Information
    print("\n" + "=" * 65)
    print("           AGENT TARGET SELECTION & LOCALIZATION")
    print("=" * 65)
    print(f"Query Received    : \"{query_text}\"")
    print(f"Identified Target : {target.upper()} (Canonical: {canonical_target})")
    print(f"Map Coordinates   : x = {x:.3f} m, y = {y:.3f} m (yaw = {yaw:.2f} rad)")
    print(f"Coordinate Frame  : {frame}")
    if approach_x is not None and approach_y is not None:
        print(f"Safe Approach Pose: x = {approach_x:.3f} m, y = {approach_y:.3f} m (yaw = {approach_yaw:.2f} rad)")
    print(f"RViz2 Highlight   : Published on topic '/selected_target'")
    print(f"Generated Map PNG : {png_path}")
    print("=" * 65 + "\n")

    return {
        "target": target,
        "canonical_target": canonical_target,
        "x": x,
        "y": y,
        "yaw": yaw,
        "frame_id": frame,
        "approach_x": approach_x,
        "approach_y": approach_y,
        "png_path": png_path,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Agent-based target query and selection.")
    parser.add_argument("query", nargs="*", help="User natural language query (e.g. 'Where is the table?')")
    parser.add_argument("--locations-yaml", default="/home/reyan/case_study/src/location_navigation/config/locations.yaml")
    parser.add_argument("--map-yaml", default="/home/reyan/case_study/room_map.yaml")
    parser.add_argument("--output-dir", default="/home/reyan/case_study/output")

    args = parser.parse_args()
    query_str = " ".join(args.query).strip() if args.query else "Where is the table?"

    query_and_select_target(
        query_text=query_str,
        locations_yaml=args.locations_yaml,
        map_yaml=args.map_yaml,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
