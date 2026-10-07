"""
location_database.py
====================
Reusable class to load locations.yaml and resolve a named target
to (x, y, yaw, frame_id).

Usage (standalone):
    db = LocationDatabase('/path/to/locations.yaml')
    pose = db.get('kitchen')
    # Returns: {'x': 8.353, 'y': -3.164, 'yaw': 0.0, 'frame_id': 'map'}

Usage (inside a ROS 2 node):
    from ament_index_python.packages import get_package_share_directory
    yaml_path = os.path.join(
        get_package_share_directory('location_navigation'), 'config', 'locations.yaml')
    db = LocationDatabase(yaml_path)
"""

import os
import yaml


class LocationNotFoundError(Exception):
    """Raised when a requested location name is not in the database."""


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
        Parameters
        ----------
        yaml_path : str
            Absolute path to the locations YAML file.
        """
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(
                f'[LocationDatabase] YAML file not found: {yaml_path}')

        with open(yaml_path, 'r', encoding='utf-8') as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, dict) or 'locations' not in data:
            raise ValueError(
                f'[LocationDatabase] {yaml_path} must contain a top-level '
                '"locations:" key.')

        self._locations: dict = data['locations']
        print(f'[LocationDatabase] Loaded {len(self._locations)} location(s): '
              f'{list(self._locations.keys())}')

    # ------------------------------------------------------------------
    def get(self, name: str) -> dict:
        """
        Return the pose dict for *name*.

        Parameters
        ----------
        name : str
            Location name (case-insensitive, strips whitespace).

        Returns
        -------
        dict with keys: x, y, yaw, frame_id

        Raises
        ------
        LocationNotFoundError
            If *name* is not in the database.
        """
        key = name.strip().lower()
        if key not in self._locations:
            available = list(self._locations.keys())
            raise LocationNotFoundError(
                f'[LocationDatabase] Unknown location: "{name}". '
                f'Available: {available}')

        entry = self._locations[key]
        return {
            'x':        float(entry['x']),
            'y':        float(entry['y']),
            'yaw':      float(entry.get('yaw', 0.0)),
            'frame_id': str(entry.get('frame_id', 'map')),
        }

    # ------------------------------------------------------------------
    def list_locations(self) -> list:
        """Return a sorted list of all known location names."""
        return sorted(self._locations.keys())

    # ------------------------------------------------------------------
    def __contains__(self, name: str) -> bool:
        return name.strip().lower() in self._locations
