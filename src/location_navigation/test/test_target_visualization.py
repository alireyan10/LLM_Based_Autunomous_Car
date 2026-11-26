#!/usr/bin/env python3
"""
Test suite for Target Visualization, Coordinate Conversion, and PNG Map Generation.
"""

import os
import sys
import hashlib
import unittest

for extra_path in [
    os.path.expanduser('~/snap/antigravity-cli/common/local/lib/python3.14/dist-packages'),
    os.path.expanduser('~/.local/lib/python3.12/site-packages'),
]:
    if os.path.isdir(extra_path) and extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from location_navigation.location_database import LocationDatabase, TARGET_ALIASES
from location_navigation.map_png_generator import MapPixelConverter, generate_map_visualization, read_pgm_p5
from location_navigation.agent_target_query import extract_target_from_query, query_and_select_target

INITIAL_YAML_SHA256 = "440144aefd2388d21a1d6bf95f84d2c75e7b31a8dd7d2f8779ba84b76e06be53"
INITIAL_PGM_SHA256 = "0f7944162f7764747c3270aa35940012fbe496c4974ade4902d17e9d039eb01e"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class TestTargetVisualization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.map_yaml = "/home/reyan/case_study/room_map.yaml"
        cls.map_pgm = "/home/reyan/case_study/room_map.pgm"
        cls.locations_yaml = "/home/reyan/case_study/src/location_navigation/config/locations.yaml"
        cls.output_dir = "/home/reyan/case_study/output"
        cls.db = LocationDatabase(cls.locations_yaml)

    def test_original_map_file_integrity(self):
        """Rule 1 & 2: room_map.pgm and room_map.yaml must remain 100% untouched."""
        curr_yaml_hash = sha256_file(self.map_yaml)
        curr_pgm_hash = sha256_file(self.map_pgm)
        self.assertEqual(curr_yaml_hash, INITIAL_YAML_SHA256, "room_map.yaml was modified!")
        self.assertEqual(curr_pgm_hash, INITIAL_PGM_SHA256, "room_map.pgm was modified!")

    def test_location_database_targets(self):
        """Rule 8: Verify actual target names and coordinates from locations.yaml."""
        locs = self.db.list_locations()
        expected = ["bed", "refrigerator", "sofa", "table"]
        self.assertEqual(sorted(locs), sorted(expected))

        # Check refrigerator
        fridge = self.db.get("refrigerator")
        self.assertAlmostEqual(fridge["x"], 8.353, places=2)
        self.assertAlmostEqual(fridge["y"], -3.165, places=2)

        # Check table
        tbl = self.db.get("table")
        self.assertAlmostEqual(tbl["x"], 5.921, places=2)
        self.assertAlmostEqual(tbl["y"], 0.612, places=2)

    def test_alias_resolution(self):
        """Verify kitchen->refrigerator, bedroom->bed, tv->sofa, dining_table->table."""
        self.assertEqual(self.db.get("kitchen")["x"], self.db.get("refrigerator")["x"])
        self.assertEqual(self.db.get("bedroom")["x"], self.db.get("bed")["x"])
        self.assertEqual(self.db.get("tv")["x"], self.db.get("sofa")["x"])
        self.assertEqual(self.db.get("dining_table")["x"], self.db.get("table")["x"])

    def test_coordinate_to_pixel_conversion(self):
        """Feature D: Coordinate conversion bounds and accuracy."""
        w, h, _ = read_pgm_p5(self.map_pgm)
        self.assertEqual(w, 376)
        self.assertEqual(h, 223)

        conv = MapPixelConverter(resolution=0.05, origin_x=-8.469, origin_y=-5.614, img_width=w, img_height=h)

        # Bottom-left origin (-8.469, -5.614) should map to u=0, v=h-1=222
        u_orig, v_orig = conv.world_to_pixel(-8.469, -5.614)
        self.assertAlmostEqual(u_orig, 0.0, places=2)
        self.assertAlmostEqual(v_orig, 222.0, places=2)

        # Check that all actual target coordinates fall within map image pixel boundaries
        for name in self.db.list_locations():
            pose = self.db.get(name)
            u, v = conv.world_to_pixel(pose["x"], pose["y"])
            self.assertTrue(0 <= u < w, f"Target {name} u={u} out of bounds [0, {w}]")
            self.assertTrue(0 <= v < h, f"Target {name} v={v} out of bounds [0, {h}]")

    def test_png_generation(self):
        """Feature C: High-resolution PNG map generation."""
        out_png = os.path.join(self.output_dir, "target_map_visualization.png")
        result = generate_map_visualization(
            map_yaml_path=self.map_yaml,
            locations_yaml_path=self.locations_yaml,
            output_png_path=out_png,
            scale_factor=4,
        )
        self.assertTrue(os.path.isfile(result))
        self.assertGreater(os.path.getsize(result), 10000)

    def test_selected_target_highlight_png(self):
        """Feature E: Selected target highlight PNG generation."""
        out_png = os.path.join(self.output_dir, "selected_target_kitchen.png")
        result = generate_map_visualization(
            map_yaml_path=self.map_yaml,
            locations_yaml_path=self.locations_yaml,
            output_png_path=out_png,
            selected_target="kitchen",
            scale_factor=4,
        )
        self.assertTrue(os.path.isfile(result))
        self.assertGreater(os.path.getsize(result), 10000)

    def test_agent_query_extraction(self):
        """Feature B: Agent natural language query extraction (English & Bangla)."""
        # English queries
        self.assertEqual(extract_target_from_query("Where is the table?", self.db), "table")
        self.assertEqual(extract_target_from_query("Where is the kitchen?", self.db), "kitchen")
        self.assertEqual(extract_target_from_query("Find the sofa", self.db), "sofa")
        self.assertEqual(extract_target_from_query("Where is bedroom?", self.db), "bedroom")

        # Bangla queries
        self.assertEqual(extract_target_from_query("টেবিল কোথায়?", self.db), "table")
        self.assertEqual(extract_target_from_query("রান্নাঘর কোন দিকে?", self.db), "kitchen")
        self.assertEqual(extract_target_from_query("সোফায় যাও", self.db), "sofa")


if __name__ == "__main__":
    unittest.main()
