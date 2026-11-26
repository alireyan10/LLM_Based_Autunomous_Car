#!/usr/bin/env python3
"""
High-Resolution PNG Map Generator for car_bot Autonomous Robot Project.

Features:
- Preserves original room_map.pgm and room_map.yaml (never modifies them).
- Accurately converts world coordinates (x, y) into image pixel coordinates (u, v).
- Renders high-resolution PNG with anti-aliasing, styled target pins, badges, and legend.
- Supports highlighting a selected target with coordinate callouts.
- Supports rendering robot start pose, current pose, approach poses, and planned Nav2 paths.
"""

import os
import sys
import math
from typing import Dict, List, Optional, Tuple, Any

# Ensure custom Python libraries (like Pillow in snap or user space) can be found
for extra_path in [
    os.path.expanduser('~/snap/antigravity-cli/common/local/lib/python3.14/dist-packages'),
    os.path.expanduser('~/.local/lib/python3.12/site-packages'),
    os.path.expanduser('~/.local/lib/python3.10/site-packages'),
]:
    if os.path.isdir(extra_path) and extra_path not in sys.path:
        sys.path.insert(0, extra_path)

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# Canonical target color mapping
TARGET_COLORS = {
    "refrigerator": {"fill": (46, 204, 113), "border": (25, 111, 61), "label": "Kitchen / Fridge"},
    "kitchen":      {"fill": (46, 204, 113), "border": (25, 111, 61), "label": "Kitchen"},
    "bed":          {"fill": (52, 152, 219), "border": (27, 79, 114), "label": "Bedroom / Bed"},
    "bedroom":      {"fill": (52, 152, 219), "border": (27, 79, 114), "label": "Bedroom"},
    "sofa":         {"fill": (230, 126, 34), "border": (120, 66, 18), "label": "Sofa / Living"},
    "tv":           {"fill": (155, 89, 182), "border": (81, 46, 95), "label": "TV Area"},
    "table":        {"fill": (231, 76, 60),  "border": (120, 40, 31), "label": "Dining Table"},
    "dining_table": {"fill": (243, 156, 18), "border": (125, 80, 9),  "label": "Dining Area"},
}

DEFAULT_TARGET_COLOR = {"fill": (52, 73, 94), "border": (23, 32, 42), "label": "Target"}


def load_yaml_fallback(file_path: str) -> dict:
    """Lightweight YAML parser for room_map.yaml and locations.yaml."""
    if HAS_YAML:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # Basic fallback parser
    data: Dict[str, Any] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                if v.startswith("[") and v.endswith("]"):
                    parts = [float(x.strip()) for x in v[1:-1].split(",") if x.strip()]
                    data[k] = parts
                else:
                    try:
                        data[k] = float(v)
                    except ValueError:
                        data[k] = v
    return data


def read_pgm_p5(file_path: str) -> Tuple[int, int, bytes]:
    """Read binary P5 PGM image and return (width, height, raw_bytes)."""
    with open(file_path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError(f"Unsupported PGM magic number: {magic}, expected P5")

        # Read header tokens, skipping comments
        tokens = []
        while len(tokens) < 3:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith(b"#") or not line:
                continue
            tokens.extend(line.split())

        width = int(tokens[0])
        height = int(tokens[1])
        maxval = int(tokens[2])
        if maxval != 255:
            raise ValueError(f"Expected maxval 255 in PGM, found {maxval}")

        data = f.read(width * height)
        if len(data) < width * height:
            raise ValueError(f"Truncated PGM file: expected {width*height} bytes, got {len(data)}")

        return width, height, data


class MapPixelConverter:
    """
    Handles mathematically rigorous coordinate conversion between
    ROS Map Frame (meters) and Image Pixel Coordinates.

    ROS Occupancy Grid standard:
      - origin = [x0, y0, yaw0] is at the bottom-left corner of the grid image.
      - Image space standard: (0, 0) is at top-left.
      - Therefore:
          u = (x_world - x0) / resolution
          v = (H - 1) - (y_world - y0) / resolution
    """
    def __init__(self, resolution: float, origin_x: float, origin_y: float, img_width: int, img_height: int, scale: int = 1):
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.img_width = int(img_width)
        self.img_height = int(img_height)
        self.scale = int(scale)

    def world_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        """Convert world (x, y) in meters to unscaled image pixel (u, v)."""
        u = (x - self.origin_x) / self.resolution
        v = (self.img_height - 1) - ((y - self.origin_y) / self.resolution)
        return u, v

    def world_to_scaled_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world (x, y) to scaled canvas pixel integers (su, sv)."""
        u, v = self.world_to_pixel(x, y)
        su = int(round(u * self.scale))
        sv = int(round(v * self.scale))
        return su, sv

    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """Inverse: pixel (u, v) back to world (x, y) in meters."""
        x = self.origin_x + (u * self.resolution)
        y = self.origin_y + ((self.img_height - 1 - v) * self.resolution)
        return x, y


def generate_map_visualization(
    map_yaml_path: str,
    locations_yaml_path: str,
    output_png_path: str,
    selected_target: Optional[str] = None,
    robot_start_pose: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0),
    robot_current_pose: Optional[Tuple[float, float, float]] = None,
    planned_path: Optional[List[Tuple[float, float]]] = None,
    scale_factor: int = 4,
) -> str:
    """
    Generates a high-resolution PNG map with all targets, approach poses,
    robot poses, path, legend, and optional selected target highlighting.

    Preserves original map files completely.
    """
    if not os.path.isfile(map_yaml_path):
        raise FileNotFoundError(f"Map YAML not found: {map_yaml_path}")
    if not os.path.isfile(locations_yaml_path):
        raise FileNotFoundError(f"Locations YAML not found: {locations_yaml_path}")

    # Load map metadata
    map_meta = load_yaml_fallback(map_yaml_path)
    res = float(map_meta.get("resolution", 0.05))
    origin = map_meta.get("origin", [-8.469, -5.614, 0.0])
    origin_x, origin_y = float(origin[0]), float(origin[1])

    # Find PGM image path relative to YAML
    pgm_filename = map_meta.get("image", "room_map.pgm")
    pgm_path = os.path.join(os.path.dirname(map_yaml_path), pgm_filename)
    if not os.path.isfile(pgm_path):
        # Fallback to direct path
        pgm_path = os.path.expanduser("~/case_study/room_map.pgm")

    # Read original PGM bytes
    w, h, pgm_bytes = read_pgm_p5(pgm_path)
    converter = MapPixelConverter(res, origin_x, origin_y, w, h, scale=scale_factor)

    # Load locations
    loc_data = load_yaml_fallback(locations_yaml_path)
    locations = loc_data.get("locations", {})

    # Create directory for output if it doesn't exist
    out_dir = os.path.dirname(os.path.abspath(output_png_path))
    os.makedirs(out_dir, exist_ok=True)

    if not HAS_PIL:
        raise RuntimeError("Pillow library is required for high-resolution PNG generation.")

    # Create base RGB image from PGM
    raw_img = Image.frombytes("L", (w, h), pgm_bytes).convert("RGB")

    # Upscale base map using Nearest Neighbor to maintain crisp obstacle contours
    canvas_w = w * scale_factor
    canvas_h = h * scale_factor

    # Add top banner and bottom legend padding
    top_pad = 70
    bottom_pad = 90
    total_w = canvas_w
    total_h = canvas_h + top_pad + bottom_pad

    scaled_map = raw_img.resize((canvas_w, canvas_h), Image.Resampling.NEAREST)

    # Build main canvas
    canvas = Image.new("RGB", (total_w, total_h), color=(245, 247, 250))
    canvas.paste(scaled_map, (0, top_pad))

    draw = ImageDraw.Draw(canvas, "RGBA")

    # Load fonts
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_subtitle = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_coord = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_legend = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font_title = ImageFont.load_default()
        font_subtitle = ImageFont.load_default()
        font_label = ImageFont.load_default()
        font_coord = ImageFont.load_default()
        font_legend = ImageFont.load_default()

    # Draw Header Banner
    draw.rectangle([0, 0, total_w, top_pad], fill=(33, 43, 54, 255))
    draw.text((20, 12), "HOUSE MAP - Target Location & Navigation Visualization", fill=(255, 255, 255), font=font_title)
    sub_text = f"Map: {w}x{h} px | Resolution: {res:.3f} m/px | Origin: [{origin_x:.3f}, {origin_y:.3f}] | Frame: map"
    draw.text((22, 44), sub_text, fill=(189, 195, 199), font=font_subtitle)

    # Normalize selected target name
    sel_norm = selected_target.strip().lower() if selected_target else None
    from location_navigation.location_database import TARGET_ALIASES
    canonical_sel = TARGET_ALIASES.get(sel_norm, sel_norm) if sel_norm else None

    # Draw Planned Path if available
    if planned_path and len(planned_path) > 1:
        path_pixels = []
        for px, py in planned_path:
            su, sv = converter.world_to_scaled_pixel(px, py)
            path_pixels.append((su, sv + top_pad))
        # Draw thick path with outline
        draw.line(path_pixels, fill=(41, 128, 185, 180), width=6)
        draw.line(path_pixels, fill=(52, 152, 219, 255), width=3)

    # Draw Targets and Approach Poses
    for target_key, data in locations.items():
        tx, ty = float(data["x"]), float(data["y"])
        su, sv = converter.world_to_scaled_pixel(tx, ty)
        sv += top_pad

        is_selected = (canonical_sel == target_key) or (sel_norm == target_key)
        color_info = TARGET_COLORS.get(target_key, DEFAULT_TARGET_COLOR)
        fill_color = color_info["fill"]
        border_color = color_info["border"]
        display_label = color_info.get("label", target_key.capitalize())

        # Draw Approach Pose if available
        if "approach_x" in data and "approach_y" in data:
            ax, ay = float(data["approach_x"]), float(data["approach_y"])
            au, av = converter.world_to_scaled_pixel(ax, ay)
            av += top_pad

            # Dashed line from approach to target
            draw.line([(au, av), (su, sv)], fill=(120, 120, 120, 180), width=2)
            # Small circle at approach pose
            ar_radius = 5
            draw.ellipse([au - ar_radius, av - ar_radius, au + ar_radius, av + ar_radius],
                         fill=(255, 255, 255, 220), outline=border_color, width=2)

        # Highlight Halo for Selected Target
        if is_selected:
            halo_r = 28
            draw.ellipse([su - halo_r, sv - halo_r, su + halo_r, sv + halo_r],
                         fill=(241, 196, 15, 90), outline=(243, 156, 18, 230), width=4)
            halo_inner = 20
            draw.ellipse([su - halo_inner, sv - halo_inner, su + halo_inner, sv + halo_inner],
                         fill=(255, 241, 118, 120), outline=(230, 126, 34, 255), width=2)

        # Draw Target Pin (Circle with distinct fill and dark border)
        pin_r = 13 if not is_selected else 16
        draw.ellipse([su - pin_r, sv - pin_r, su + pin_r, sv + pin_r],
                     fill=fill_color + (240,), outline=border_color + (255,), width=3)
        # Inner white center dot
        inner_r = 4 if not is_selected else 5
        draw.ellipse([su - inner_r, sv - inner_r, su + inner_r, sv + inner_r],
                     fill=(255, 255, 255, 255))

        # Target Label Badge (with background for high contrast)
        badge_title = display_label if not is_selected else f"★ {display_label} (SELECTED)"
        badge_coord = f"({tx:.2f}, {ty:.2f})"

        # Determine badge placement (offset slightly to avoid overlapping pin)
        bx = su + pin_r + 6
        by = sv - 16
        if bx + 160 > total_w:
            bx = su - pin_r - 160  # Flip to left if near right edge

        # Measure text box
        tb_w = 150 if is_selected else 125
        tb_h = 36 if is_selected else 30
        badge_bg = (241, 196, 15, 235) if is_selected else (255, 255, 255, 225)
        badge_border = (214, 137, 16) if is_selected else (149, 165, 166)

        # Draw rounded badge rectangle
        draw.rounded_rectangle([bx, by, bx + tb_w, by + tb_h], radius=6, fill=badge_bg, outline=badge_border, width=2)
        title_fill = (20, 20, 20) if not is_selected else (100, 40, 0)
        draw.text((bx + 8, by + 3), badge_title, fill=title_fill, font=font_label)
        draw.text((bx + 8, by + 18), badge_coord, fill=(80, 80, 80), font=font_coord)

    # Draw Robot Start Position (0.0, 0.0)
    if robot_start_pose is not None:
        sx, sy, syaw = robot_start_pose
        rsu, rsv = converter.world_to_scaled_pixel(sx, sy)
        rsv += top_pad
        # Draw Robot Start Icon (Cyan Diamond / Circle)
        draw.ellipse([rsu - 10, rsv - 10, rsu + 10, rsv + 10], fill=(26, 188, 156, 230), outline=(19, 141, 117), width=3)
        # Orientation Arrow
        arrow_len = 16
        ax_end = rsu + arrow_len * math.cos(syaw)
        ay_end = rsv - arrow_len * math.sin(syaw)  # inverted y in image
        draw.line([(rsu, rsv), (ax_end, ay_end)], fill=(22, 160, 133), width=3)
        draw.text((rsu + 14, rsv - 8), "Robot Start (0, 0)", fill=(19, 141, 117), font=font_label)

    # Draw Robot Current Position if available
    if robot_current_pose is not None:
        cx, cy, cyaw = robot_current_pose
        rcu, rcv = converter.world_to_scaled_pixel(cx, cy)
        rcv += top_pad
        draw.ellipse([rcu - 12, rcv - 12, rcu + 12, rcv + 12], fill=(231, 76, 60, 230), outline=(146, 43, 33), width=3)
        c_arrow_len = 18
        c_end_x = rcu + c_arrow_len * math.cos(cyaw)
        c_end_y = rcv - c_arrow_len * math.sin(cyaw)
        draw.line([(rcu, rcv), (c_end_x, c_end_y)], fill=(255, 255, 255), width=3)
        draw.text((rcu + 16, rcv - 8), "Robot Current", fill=(146, 43, 33), font=font_label)

    # Draw Footer & Legend
    footer_y = canvas_h + top_pad
    draw.rectangle([0, footer_y, total_w, total_h], fill=(236, 240, 241, 255))
    draw.line([(0, footer_y), (total_w, footer_y)], fill=(189, 195, 199), width=2)

    draw.text((20, footer_y + 12), "LEGEND:", fill=(44, 62, 80), font=font_label)

    legend_items = [
        ("Kitchen / Fridge", (46, 204, 113)),
        ("Bedroom / Bed", (52, 152, 219)),
        ("Sofa / Living", (230, 126, 34)),
        ("Dining Table", (231, 76, 60)),
        ("Robot Start", (26, 188, 156)),
        ("Approach Pose (o)", (149, 165, 166)),
    ]
    if planned_path:
        legend_items.append(("Nav2 Path", (52, 152, 219)))
    if selected_target:
        legend_items.append(("Selected (★)", (241, 196, 15)))

    cur_x = 100
    for label, col in legend_items:
        draw.ellipse([cur_x, footer_y + 16, cur_x + 12, footer_y + 28], fill=col, outline=(50, 50, 50), width=1)
        draw.text((cur_x + 18, footer_y + 14), label, fill=(44, 62, 80), font=font_legend)
        cur_x += 160

    footer_note = "Generated from room_map.pgm & locations.yaml without modifying source map files."
    draw.text((20, footer_y + 55), footer_note, fill=(127, 140, 141), font=font_coord)

    # Save PNG
    canvas.save(output_png_path, "PNG", optimize=True)
    print(f"[MapPNGGenerator] Successfully generated high-resolution map: {output_png_path} ({total_w}x{total_h})")
    return output_png_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate high-resolution PNG visualization of robot room map with target locations.")
    parser.add_argument("--map-yaml", default="/home/reyan/case_study/room_map.yaml", help="Path to room_map.yaml")
    parser.add_argument("--locations-yaml", default="/home/reyan/case_study/src/location_navigation/config/locations.yaml", help="Path to locations.yaml")
    parser.add_argument("--output", default="/home/reyan/case_study/output/target_map_visualization.png", help="Path for output PNG")
    parser.add_argument("--select", default=None, help="Target name to highlight (e.g. kitchen, table, sofa, bed)")
    parser.add_argument("--scale", type=int, default=4, help="Canvas scale factor (default: 4)")

    args = parser.parse_args()

    # If select is specified and output is default, also generate selected_target_<name>.png
    out_path = args.output
    if args.select and out_path.endswith("target_map_visualization.png"):
        sel_clean = args.select.strip().lower().replace(" ", "_")
        out_path = os.path.join(os.path.dirname(out_path), f"selected_target_{sel_clean}.png")

    generate_map_visualization(
        map_yaml_path=args.map_yaml,
        locations_yaml_path=args.locations_yaml,
        output_png_path=out_path,
        selected_target=args.select,
        scale_factor=args.scale,
    )


if __name__ == "__main__":
    main()
