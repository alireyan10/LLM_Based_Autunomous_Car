
import os
from typing import Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None


class LocationNotFoundError(Exception):
    """Raised when a requested location name is not in the database."""


TARGET_ALIASES: Dict[str, str] = {
    "bedroom": "bed",
    "kitchen": "refrigerator",
    "tv": "sofa",
    "dining_table": "table",
}


class LocationDatabase:
    """
    Loads a locations.yaml file and provides lookup by name.

    Expected YAML format::

        locations:
          kitchen:
            x: 8.353
            y: -3.164
            yaw: 0.0
            frame_id: map
    """

    def __init__(self, yaml_path: str):
        """
 load  locations.yaml file  location and store it in data memory .
 If PyYAML is unavailable, it manually reads locations.yaml and converts the location data into a dictionary.

        """

        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(
                f'[LocationDatabase] YAML file not found: {yaml_path}')

        data = None
        if yaml is not None:
            try:
                with open(yaml_path, 'r', encoding='utf-8') as fh:
                    data = yaml.safe_load(fh)
            except Exception as exc:
                data = None

        if data is None or not isinstance(data, dict) or 'locations' not in data:
            # Fallback lightweight parser when PyYAML is unavailable
            data = self._parse_locations_fallback(yaml_path)

        if not isinstance(data, dict) or 'locations' not in data:
            raise ValueError(
                f'[LocationDatabase] {yaml_path} must contain a top-level '
                '"locations:" key.')

        self._locations: dict = data['locations']
        print(f'[LocationDatabase] Loaded {len(self._locations)} location(s): '
              f'{list(self._locations.keys())}')


    def _parse_locations_fallback(self, yaml_path: str) -> dict:
        """Lightweight parser for locations.yaml when PyYAML is not installed."""

        locations = {}
        current_loc = None

        with open(yaml_path, 'r', encoding='utf-8') as fh:
            for line in fh:
                raw = line.rstrip()
                stripped = raw.strip()
                if not stripped or stripped.startswith('#') or stripped == 'locations:':
                    continue

                if raw.startswith('  ') and not raw.startswith('    ') and stripped.endswith(':'):
                    current_loc = stripped[:-1].strip().lower()
                    locations[current_loc] = {}
                elif raw.startswith('    ') and current_loc and ':' in stripped:
                    key, val = stripped.split(':', 1)
                    key = key.strip()
                    val = val.strip()
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                    locations[current_loc][key] = val

        return {'locations': locations}


    def get(self, name: str) -> dict:
        """
Gets the target pose from a location name or alias.
Resolves aliases, such as bedroom -> bed.
Returns x, y, yaw, and frame_id.
Raises LocationNotFoundError if the location is not found.

        """
        raw_key = name.strip().lower()
        key = TARGET_ALIASES.get(raw_key, raw_key)
        if key not in self._locations:
            available = list(self._locations.keys())
            raise LocationNotFoundError(
                f'[LocationDatabase] Unknown location: "{name}". '
                f'Available: {available}')

        entry = self._locations[key]
        pose = {
            'x':        float(entry['x']),
            'y':        float(entry['y']),
            'yaw':      float(entry.get('yaw', 0.0)),
            'frame_id': str(entry.get('frame_id', 'map')),
        }
        for opt_field in (
            'approach_x',
            'approach_y',
            'approach_yaw',
            'approach_direction',
            'standoff_distance',
        ):
            if opt_field in entry and entry[opt_field] is not None:
                try:
                    pose[opt_field] = float(entry[opt_field])
                except (ValueError, TypeError):
                    pass
        return pose


    def get_approach_pose(
        self, name: str, standoff: Optional[float] = None
    ) -> Optional[dict]:
        """
Calculates the approach pose for the robot to stop near the target.
Uses the configured approach direction or position, if available.
Calculates the yaw so the robot faces the target.
Returns None if a safe approach pose cannot be found.

        """

        import math
        pose = self.get(name)
        x_target = pose['x']
        y_target = pose['y']
        frame_id = pose.get('frame_id', 'map')

        dist = standoff
        if dist is None or dist <= 0.0:
            dist = pose.get('standoff_distance', 0.8)

        # Priority 1: Configured approach direction
        if 'approach_direction' in pose:
            angle = float(pose['approach_direction'])
            ax = x_target + dist * math.cos(angle)
            ay = y_target + dist * math.sin(angle)
            if 'approach_yaw' in pose:
                ayaw = float(pose['approach_yaw'])
            else:
                ayaw = math.atan2(y_target - ay, x_target - ax)
            return {'x': ax, 'y': ay, 'yaw': ayaw, 'frame_id': frame_id}

        # Priority 2: Configured explicit approach position
        if 'approach_x' in pose and 'approach_y' in pose:
            ax = float(pose['approach_x'])
            ay = float(pose['approach_y'])
            if standoff is not None and standoff > 0.0:
                dx = ax - x_target
                dy = ay - y_target
                cur_dist = math.hypot(dx, dy)
                if cur_dist > 1e-4:
                    scale = standoff / cur_dist
                    ax = x_target + dx * scale
                    ay = y_target + dy * scale
            if 'approach_yaw' in pose:
                ayaw = float(pose['approach_yaw'])
            else:
                ayaw = math.atan2(y_target - ay, x_target - ax)
            return {'x': ax, 'y': ay, 'yaw': ayaw, 'frame_id': frame_id}

        return None


    def list_locations(self) -> list:
        """Return a sorted list of all known canonical location names."""
        return sorted(self._locations.keys())

    @property
    def aliases(self) -> Dict[str, str]:
        """Return a copy of target aliases."""
        return dict(TARGET_ALIASES)


    def __contains__(self, name: str) -> bool:
        #Checks whether a location or alias exists in the database.

        raw_key = name.strip().lower()
        key = TARGET_ALIASES.get(raw_key, raw_key)
        return key in self._locations

