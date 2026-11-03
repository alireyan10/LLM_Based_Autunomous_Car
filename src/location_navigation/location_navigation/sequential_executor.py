#!/usr/bin/env python3
"""
Validates the action plan structure, intents, parameters, and target locations.
Executes multiple robot actions in the specified order.
Tracks the current action, execution status, and completed steps.
Stops the sequence if an action fails or a software STOP request is received.
Connects actions to the ROS 2 backend; it does not directly control motors or provide hardware emergency stop.
"""

import json
import logging
import math
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

try:
    import yaml
except ImportError:
    yaml = None


# ---------------------------------------------------------------------------
# Supported Actions and Constants
# ---------------------------------------------------------------------------
LOCATION_INTENTS: Set[str] = {
    "MOVE_TO_TARGET",
    "MOVE_NEAR_TARGET",
    "MOVE_TOWARD_TARGET",
}

MOTION_INTENTS: Set[str] = {
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
}

SPECIAL_INTENTS: Set[str] = {
    "STOP",
}

SUPPORTED_INTENTS: Set[str] = LOCATION_INTENTS | MOTION_INTENTS | SPECIAL_INTENTS

TOP_LEVEL_INTENTS: Set[str] = SUPPORTED_INTENTS | {"SEQUENTIAL"}

# Action aliases mapping to canonical intents
ACTION_ALIASES: Dict[str, str] = {
    "ROTATE_LEFT": "TURN_LEFT",
    "ROTATE_RIGHT": "TURN_RIGHT",
}

# Safe Parameter Defaults (Applied only to omitted parameters)
DEFAULT_FORWARD_DISTANCE: float = 1.0       # meters
DEFAULT_BACKWARD_DISTANCE: float = 1.0      # meters
DEFAULT_TURN_ANGLE_DEG: float = 90.0        # degrees

# Parameter Maximum Limits
MAX_MOVEMENT_DISTANCE: float = 2.0          # meters per movement action
MIN_MOVEMENT_DISTANCE: float = 0.01         # meters
MAX_TURN_ANGLE_DEG: float = 180.0           # degrees absolute turn angle



# ---------------------------------------------------------------------------
# Execution Status
# ---------------------------------------------------------------------------
class ExecutionStatus(str, Enum):
    """Lifecycle status of the action executor."""
    IDLE = "IDLE"
    VALIDATING = "VALIDATING"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Result Data Structures
# ---------------------------------------------------------------------------
@dataclass
class PlanValidationResult:
    """Result of validating an action plan."""
    is_valid: bool
    error: Optional[str] = None
    normalized_plan: Optional[Dict[str, Any]] = None

    def __bool__(self) -> bool:
        return self.is_valid


@dataclass
class ExecutionResult:
    """Result of an action sequence execution attempt."""
    success: bool
    status: ExecutionStatus
    message: str
    completed_steps: int = 0
    total_steps: int = 0
    failed_step_index: Optional[int] = None
    execution_history: List[Dict[str, Any]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.success


# ---------------------------------------------------------------------------
# Execution Backend Interface
# ---------------------------------------------------------------------------
class ActionExecutionBackend(ABC):
    """
    Abstract execution backend interface.

    Concrete implementations will interface with ROS 2 Nav2 (NavigateToPose)
    and a safe motion controller. This interface decouples high-level plan
    sequencing and validation from low-level robot actions.
    """

    @abstractmethod
    def execute_action(self, action: Dict[str, Any]) -> bool:
        """
        Execute a single validated action dictionary.

        Parameters
        ----------
        action : dict
            Normalized action dictionary, e.g. {'intent': 'MOVE_FORWARD', 'distance': 1.0}

        Returns
        -------
        bool
            True if action execution succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def cancel_or_stop(self) -> bool:
        """
        Request cancellation of active navigation goals or motion.

        NOTE: This is a software cancel request, NOT a hardware emergency stop.

        Returns
        -------
        bool
            True if cancel/stop request was dispatched successfully, False otherwise.
        """
        pass


# ---------------------------------------------------------------------------
# Helper: Load Locations from YAML
# ---------------------------------------------------------------------------
def load_locations_from_yaml(yaml_path: str) -> Set[str]:
    """
    Safely load known location names from a YAML file.

    Parameters
    ----------
    yaml_path : str
        Absolute or relative path to locations.yaml.

    Returns
    -------
    set of str
        Set of lowercase location names found in the database.
    """
    if not os.path.isfile(yaml_path):
        return set()

    # Attempt parsing with PyYAML if available
    if yaml is not None:
        try:
            with open(yaml_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if isinstance(data, dict) and "locations" in data and isinstance(data["locations"], dict):
                return {str(k).strip().lower() for k in data["locations"].keys()}
        except Exception as exc:
            logging.getLogger(__name__).warning(
                f"[SequentialActionExecutor] Failed to parse {yaml_path} with PyYAML: {exc}"
            )

    # Lightweight fallback parser if PyYAML is unavailable
    locations: Set[str] = set()
    try:
        in_locations = False
        with open(yaml_path, "r", encoding="utf-8") as fh:
            for line in fh:
                raw_line = line.rstrip()
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("locations:"):
                    in_locations = True
                    continue
                if in_locations:
                    if raw_line.startswith("  ") and not raw_line.startswith("    ") and stripped.endswith(":"):
                        loc_name = stripped[:-1].strip().lower()
                        if loc_name:
                            locations.add(loc_name)
                    elif not raw_line.startswith(" ") and not raw_line.startswith("\t"):
                        break
        return locations
    except Exception as exc:
        logging.getLogger(__name__).warning(
            f"[SequentialActionExecutor] Fallback YAML parser error on {yaml_path}: {exc}"
        )
        return set()


# ---------------------------------------------------------------------------
# SequentialActionExecutor Class
# ---------------------------------------------------------------------------
class SequentialActionExecutor:
    """
    Sequential action execution manager for ROS 2 Jazzy mobile robot.

    Responsible for:
    - Validating structured action plans before execution.
    - Processing actions in strict sequence order.
    - Tracking the current action index and overall execution status.
    - Halting sequence execution immediately if an action fails.
    - Safely handling STOP requests as software cancellations.
    - Enforcing backend connection before executing actions (no false successes).
    """

    def __init__(
        self,
        known_locations: Optional[Iterable[str]] = None,
        backend: Optional[ActionExecutionBackend] = None,
        logger: Optional[Any] = None,
        locations_yaml_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the executor.

        Parameters
        ----------
        known_locations : Iterable[str], optional
            Known valid location targets. If None, executor attempts to load
            locations from locations.yaml.
        backend : ActionExecutionBackend, optional
            Execution backend for dispatching actions. If None, execution is
            marked unavailable.
        logger : Any, optional
            ROS 2 Node logger (node.get_logger()) or standard Python logger.
        locations_yaml_path : str, optional
            Path to locations.yaml file to load locations from.
        """
        self._logger = logger
        self._backend: Optional[ActionExecutionBackend] = backend
        self._action_handlers: Dict[str, Callable[[Dict[str, Any]], bool]] = {}
        self._stop_handler: Optional[Callable[[], bool]] = None

        # Tracking state
        self._status: ExecutionStatus = ExecutionStatus.IDLE
        self._current_action_index: int = -1
        self._current_action: Optional[Dict[str, Any]] = None
        self._execution_history: List[Dict[str, Any]] = []
        self._cancel_requested: bool = False

        # Known locations database
        self._known_locations: Set[str] = set()
        if known_locations is not None:
            self.set_known_locations(known_locations)
        elif locations_yaml_path is not None:
            self._known_locations = load_locations_from_yaml(locations_yaml_path)
        else:
            self._discover_default_locations()

    # ------------------------------------------------------------------
    # Logging Helpers (Compatible with rclpy logger and standard logger)
    # ------------------------------------------------------------------
    def _log_info(self, msg: str) -> None:
        if self._logger is not None and hasattr(self._logger, "info"):
            self._logger.info(msg)
        else:
            logging.info(f"[SequentialActionExecutor] {msg}")

    def _log_warn(self, msg: str) -> None:
        if self._logger is not None and hasattr(self._logger, "warning"):
            self._logger.warning(msg)
        elif self._logger is not None and hasattr(self._logger, "warn"):
            self._logger.warn(msg)
        else:
            logging.warning(f"[SequentialActionExecutor] {msg}")

    def _log_error(self, msg: str) -> None:
        if self._logger is not None and hasattr(self._logger, "error"):
            self._logger.error(msg)
        else:
            logging.error(f"[SequentialActionExecutor] {msg}")

    # ------------------------------------------------------------------
    # Location Management
    # ------------------------------------------------------------------
    def _discover_default_locations(self) -> None:
        """Attempt to discover locations.yaml in default project locations."""
        # Check relative path: ../config/locations.yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        rel_path = os.path.normpath(os.path.join(current_dir, "..", "config", "locations.yaml"))
        if os.path.isfile(rel_path):
            self._known_locations = load_locations_from_yaml(rel_path)
            return

        # Check ROS 2 package share directory if in a sourced ROS 2 environment
        try:
            from ament_index_python.packages import get_package_share_directory
            pkg_share = get_package_share_directory("location_navigation")
            share_yaml = os.path.join(pkg_share, "config", "locations.yaml")
            if os.path.isfile(share_yaml):
                self._known_locations = load_locations_from_yaml(share_yaml)
        except Exception:
            pass

    def set_known_locations(self, locations: Iterable[str]) -> None:
        """Set the collection of known location names."""
        self._known_locations = {str(loc).strip().lower() for loc in locations if str(loc).strip()}

    def add_known_location(self, location_name: str) -> None:
        """Add a single known location name."""
        name = location_name.strip().lower()
        if name:
            self._known_locations.add(name)

    def is_known_location(self, location_name: str) -> bool:
        """Check whether a target location is known."""
        name = location_name.strip().lower()
        if not name or name in {"unknown", "unknown_target"}:
            return False
        if not self._known_locations:
            # If no known location database is loaded, reject empty/unknown keywords
            return True
        # Check canonical or legacy alias
        legacy_aliases = {"kitchen": "refrigerator", "bedroom": "bed", "tv": "sofa", "dining_table": "table"}
        canonical = legacy_aliases.get(name, name)
        return (name in self._known_locations) or (canonical in self._known_locations)

    @property
    def known_locations(self) -> Set[str]:
        """Return a copy of the known locations set."""
        return set(self._known_locations)

    # ------------------------------------------------------------------
    # Backend Management
    # ------------------------------------------------------------------
    def set_backend(self, backend: ActionExecutionBackend) -> None:
        """Connect an execution backend."""
        self._backend = backend

    def register_action_handler(
        self, intent: str, handler: Callable[[Dict[str, Any]], bool]
    ) -> None:
        """Register a specific execution callback handler for an action intent."""
        self._action_handlers[intent.strip().upper()] = handler

    def register_stop_handler(self, handler: Callable[[], bool]) -> None:
        """Register a software stop/cancel callback handler."""
        self._stop_handler = handler

    def is_backend_connected(self) -> bool:
        """Check whether an execution backend or action handlers are connected."""
        return self._backend is not None or bool(self._action_handlers)

    # ------------------------------------------------------------------
    # Status and State Inspection
    # ------------------------------------------------------------------
    @property
    def status(self) -> ExecutionStatus:
        """Current execution lifecycle status."""
        return self._status

    @property
    def current_action_index(self) -> int:
        """0-indexed index of currently active action, or -1 when idle."""
        return self._current_action_index

    @property
    def current_action(self) -> Optional[Dict[str, Any]]:
        """Dictionary of currently executing action, or None."""
        return self._current_action

    @property
    def execution_history(self) -> List[Dict[str, Any]]:
        """History of executed steps in the most recent execution attempt."""
        return list(self._execution_history)

    def reset(self) -> None:
        """Reset execution state back to IDLE."""
        self._status = ExecutionStatus.IDLE
        self._current_action_index = -1
        self._current_action = None
        self._execution_history.clear()
        self._cancel_requested = False

    # ------------------------------------------------------------------
    # Plan Validation
    # ------------------------------------------------------------------
    def validate_plan(
        self, plan_data: Union[Dict[str, Any], str]
    ) -> PlanValidationResult:
        """
        Validate an action plan before execution.

        Parameters
        ----------
        plan_data : dict or str
            Action plan dictionary or JSON string.

        Returns
        -------
        PlanValidationResult
            Validation result containing is_valid, error message if any,
            and normalized plan structure.
        """
        # 1. Parse JSON if passed as string
        if isinstance(plan_data, str):
            try:
                plan = json.loads(plan_data)
            except json.JSONDecodeError as exc:
                err = f"Plan JSON decoding failed: {exc}"
                self._log_error(err)
                return PlanValidationResult(is_valid=False, error=err)
        elif isinstance(plan_data, dict):
            plan = plan_data
        else:
            err = f"Plan must be a dict or JSON string, got {type(plan_data).__name__}"
            self._log_error(err)
            return PlanValidationResult(is_valid=False, error=err)

        # 2. Check top-level intent
        if "intent" not in plan:
            err = "Action plan missing required field 'intent'."
            self._log_error(err)
            return PlanValidationResult(is_valid=False, error=err)

        top_intent = plan["intent"]
        if not isinstance(top_intent, str) or not top_intent.strip():
            err = "Action plan 'intent' must be a non-empty string."
            self._log_error(err)
            return PlanValidationResult(is_valid=False, error=err)

        top_intent_clean = top_intent.strip().upper()
        if top_intent_clean not in TOP_LEVEL_INTENTS:
            err = (
                f"Unsupported plan intent '{top_intent}'. "
                f"Expected 'SEQUENTIAL' or one of {sorted(SUPPORTED_INTENTS)}."
            )
            self._log_error(err)
            return PlanValidationResult(is_valid=False, error=err)

        # 3. Check action_sequence (or normalize single action dict)
        if "action_sequence" not in plan:
            if top_intent_clean in SUPPORTED_INTENTS:
                # Normalize standalone single-action dict into a 1-step sequence
                plan = {
                    "intent": "SEQUENTIAL",
                    "step_count": 1,
                    "action_sequence": [plan],
                }
                top_intent_clean = "SEQUENTIAL"
            else:
                err = "Action plan missing required field 'action_sequence'."
                self._log_error(err)
                return PlanValidationResult(is_valid=False, error=err)
        elif top_intent_clean in SUPPORTED_INTENTS:
            top_intent_clean = "SEQUENTIAL"

        action_seq = plan["action_sequence"]
        if not isinstance(action_seq, list):
            err = f"Field 'action_sequence' must be a list, got {type(action_seq).__name__}."
            self._log_error(err)
            return PlanValidationResult(is_valid=False, error=err)

        if len(action_seq) == 0:
            err = "Field 'action_sequence' must not be empty."
            self._log_error(err)
            return PlanValidationResult(is_valid=False, error=err)

        # 4. Check step_count (recalculate or validate)
        # Safe Policy: If step_count is omitted, it will be recalculated from action_sequence.
        # If explicitly provided, validate it matches len(action_sequence).
        if "step_count" in plan and plan["step_count"] is not None:
            step_count = plan["step_count"]
            if isinstance(step_count, bool) or not isinstance(step_count, int):
                err = f"Field 'step_count' must be an integer, got {type(step_count).__name__}."
                self._log_error(err)
                return PlanValidationResult(is_valid=False, error=err)
            if step_count <= 0:
                err = f"Field 'step_count' must be a positive integer, got {step_count}."
                self._log_error(err)
                return PlanValidationResult(is_valid=False, error=err)
            if step_count != len(action_seq):
                err = (
                    f"step_count ({step_count}) does not match the number of actions "
                    f"in action_sequence ({len(action_seq)})."
                )
                self._log_error(err)
                return PlanValidationResult(is_valid=False, error=err)

        # 5. Validate and normalize each individual action
        normalized_actions: List[Dict[str, Any]] = []
        for idx, raw_action in enumerate(action_seq):
            valid, norm_action, act_err = self._validate_and_normalize_action(idx, raw_action)
            if not valid:
                self._log_error(act_err)
                return PlanValidationResult(is_valid=False, error=act_err)
            normalized_actions.append(norm_action)

        # Recalculate step_count from the final normalized action_sequence
        normalized_plan = {
            "intent": top_intent_clean,
            "step_count": len(normalized_actions),
            "action_sequence": normalized_actions,
        }

        self._log_info(
            f"Action plan validated successfully: {len(normalized_actions)} step(s)."
        )
        return PlanValidationResult(is_valid=True, normalized_plan=normalized_plan)

    def _validate_and_normalize_action(
        self, idx: int, raw_action: Any
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Validate and normalize a single action item into the strictly validated schema.

        Enforces:
        - Every action must be a structured JSON object (dict), never a bare string.
        - Defaults applied only to omitted parameters:
          * MOVE_FORWARD distance: 1.0 m
          * MOVE_BACKWARD distance: 1.0 m
          * TURN_LEFT angle: 90.0 deg
          * TURN_RIGHT angle: 90.0 deg
        - Parameter maximum limits:
          * Movement distance: max 2.0 m
          * Absolute turn angle: max 180.0 deg
        - Explicit invalid values (negative, zero, non-numeric, > max limits) are rejected.
        - STOP requires no distance or angle.
        - MOVE_TO_TARGET requires a known target from locations.yaml.
        """
        step_label = f"Step {idx + 1}"

        # Requirement 2: Every action must be a structured JSON object, never a bare string.
        if isinstance(raw_action, str):
            return (
                False,
                None,
                f"{step_label}: Action must be a structured JSON object, never a bare string ('{raw_action}').",
            )

        if not isinstance(raw_action, dict):
            return (
                False,
                None,
                f"{step_label}: Action must be a structured JSON object (dict), got {type(raw_action).__name__}.",
            )

        if "intent" not in raw_action:
            return False, None, f"{step_label}: Action dictionary missing required 'intent' key."

        intent_val = raw_action["intent"]
        if not isinstance(intent_val, str) or not intent_val.strip():
            return False, None, f"{step_label}: Action 'intent' must be a non-empty string."

        raw_upper = intent_val.strip().upper()
        intent = ACTION_ALIASES.get(raw_upper, raw_upper)
        if intent not in SUPPORTED_INTENTS:
            return (
                False,
                None,
                f"{step_label}: Unsupported action intent '{intent_val}'. "
                f"Supported intents: {sorted(SUPPORTED_INTENTS)}.",
            )

        # --------------------------------------------------------------
        # Parameter Check: MOVE_FORWARD
        # --------------------------------------------------------------
        if intent == "MOVE_FORWARD":
            if "distance" not in raw_action or raw_action["distance"] is None:
                # Default omitted parameter (1.0m)
                dist = DEFAULT_FORWARD_DISTANCE
            else:
                dist = raw_action["distance"]
                if isinstance(dist, bool) or not isinstance(dist, (int, float)):
                    return (
                        False,
                        None,
                        f"{step_label}: MOVE_FORWARD 'distance' must be numeric (int or float), "
                        f"got {type(dist).__name__}.",
                    )
                if dist <= 0 or not math.isfinite(dist):
                    return (
                        False,
                        None,
                        f"{step_label}: MOVE_FORWARD 'distance' must be positive, got {dist}.",
                    )
                if dist > MAX_MOVEMENT_DISTANCE:
                    return (
                        False,
                        None,
                        f"{step_label}: MOVE_FORWARD 'distance' {dist}m exceeds maximum safety limit {MAX_MOVEMENT_DISTANCE}m.",
                    )
            return True, {"intent": "MOVE_FORWARD", "distance": float(dist)}, ""

        # --------------------------------------------------------------
        # Parameter Check: MOVE_BACKWARD
        # --------------------------------------------------------------
        if intent == "MOVE_BACKWARD":
            if "distance" not in raw_action or raw_action["distance"] is None:
                # Default omitted parameter (1.0m)
                dist = DEFAULT_BACKWARD_DISTANCE
            else:
                dist = raw_action["distance"]
                if isinstance(dist, bool) or not isinstance(dist, (int, float)):
                    return (
                        False,
                        None,
                        f"{step_label}: MOVE_BACKWARD 'distance' must be numeric (int or float), "
                        f"got {type(dist).__name__}.",
                    )
                if dist <= 0 or not math.isfinite(dist):
                    return (
                        False,
                        None,
                        f"{step_label}: MOVE_BACKWARD 'distance' must be positive, got {dist}.",
                    )
                if dist > MAX_MOVEMENT_DISTANCE:
                    return (
                        False,
                        None,
                        f"{step_label}: MOVE_BACKWARD 'distance' {dist}m exceeds maximum safety limit {MAX_MOVEMENT_DISTANCE}m.",
                    )
            return True, {"intent": "MOVE_BACKWARD", "distance": float(dist)}, ""

        # --------------------------------------------------------------
        # Parameter Check: TURN_LEFT / TURN_RIGHT
        # --------------------------------------------------------------
        if intent in {"TURN_LEFT", "TURN_RIGHT"}:
            if "angle" not in raw_action or raw_action["angle"] is None:
                # Default omitted parameter (90.0 deg)
                ang = DEFAULT_TURN_ANGLE_DEG
                unit = "deg"
            else:
                ang = raw_action["angle"]
                if isinstance(ang, bool) or not isinstance(ang, (int, float)):
                    return (
                        False,
                        None,
                        f"{step_label}: {intent} 'angle' must be numeric (int or float), "
                        f"got {type(ang).__name__}.",
                    )
                if ang <= 0 or not math.isfinite(ang):
                    return (
                        False,
                        None,
                        f"{step_label}: {intent} 'angle' must be positive, got {ang}.",
                    )
                unit = str(raw_action.get("unit", "deg")).strip().lower()
                if unit in {"rad", "radians", "radian"}:
                    if ang > (math.pi + 1e-4):
                        return (
                            False,
                            None,
                            f"{step_label}: {intent} 'angle' {ang} rad exceeds maximum safety limit of pi rad (180 deg).",
                        )
                else:
                    unit = "deg"
                    if ang > MAX_TURN_ANGLE_DEG:
                        return (
                            False,
                            None,
                            f"{step_label}: {intent} 'angle' {ang} deg exceeds maximum safety limit {MAX_TURN_ANGLE_DEG} deg.",
                        )
            norm = {"intent": intent, "angle": float(ang)}
            if unit != "deg":
                norm["unit"] = unit
            return True, norm, ""

        # --------------------------------------------------------------
        # Parameter Check: Location-based actions (MOVE_TO_TARGET)
        # --------------------------------------------------------------
        if intent in LOCATION_INTENTS:
            if "target" not in raw_action or raw_action["target"] is None:
                return (
                    False,
                    None,
                    f"{step_label}: {intent} requires a 'target' parameter. Missing parameters must not be guessed.",
                )
            target = raw_action["target"]
            if not isinstance(target, str) or not target.strip():
                return (
                    False,
                    None,
                    f"{step_label}: {intent} 'target' must be a non-empty string.",
                )
            clean_target = target.strip().lower()
            if not self.is_known_location(clean_target):
                return (
                    False,
                    None,
                    f"{step_label}: Target '{target}' is unknown or invalid. Never invent unknown target names.",
                )
            norm_action = {"intent": intent, "target": clean_target}
            if intent == "MOVE_NEAR_TARGET":
                standoff = raw_action.get("standoff", raw_action.get("standoff_distance", raw_action.get("distance")))
                if standoff is not None:
                    if isinstance(standoff, bool) or not isinstance(standoff, (int, float)):
                        return (
                            False,
                            None,
                            f"{step_label}: {intent} 'standoff' must be numeric (int or float), got {type(standoff).__name__}.",
                        )
                    if standoff <= 0 or not math.isfinite(standoff):
                        return (
                            False,
                            None,
                            f"{step_label}: {intent} 'standoff' must be positive, got {standoff}.",
                        )
                    if standoff > MAX_MOVEMENT_DISTANCE:
                        return (
                            False,
                            None,
                            f"{step_label}: {intent} 'standoff' {standoff}m exceeds maximum safety limit {MAX_MOVEMENT_DISTANCE}m.",
                        )
                    norm_action["standoff"] = float(standoff)

                app_dir = raw_action.get("approach_direction", raw_action.get("approach_yaw"))
                if app_dir is not None:
                    if isinstance(app_dir, bool) or not isinstance(app_dir, (int, float)):
                        return (
                            False,
                            None,
                            f"{step_label}: {intent} 'approach_direction' must be numeric, got {type(app_dir).__name__}.",
                        )
                    if not math.isfinite(app_dir):
                        return (
                            False,
                            None,
                            f"{step_label}: {intent} 'approach_direction' must be finite, got {app_dir}.",
                        )
                    norm_action["approach_direction"] = float(app_dir)

            return True, norm_action, ""

        # --------------------------------------------------------------
        # Parameter Check: STOP (Requires no distance or angle)
        # --------------------------------------------------------------
        if intent == "STOP":
            return True, {"intent": "STOP"}, ""

        return False, None, f"{step_label}: Unhandled intent '{intent}'."


    # ------------------------------------------------------------------
    # Sequential Execution
    # ------------------------------------------------------------------
    def execute_plan(
        self, plan_data: Union[Dict[str, Any], str]
    ) -> ExecutionResult:
        """
        Validate and execute an action sequence in strict sequential order.

        Parameters
        ----------
        plan_data : dict or str
            Action plan dictionary or JSON string.

        Returns
        -------
        ExecutionResult
            Outcome of execution, including completed steps and detailed status.
        """
        # 1. Plan Validation
        self._status = ExecutionStatus.VALIDATING
        val_result = self.validate_plan(plan_data)
        if not val_result.is_valid:
            self._status = ExecutionStatus.FAILED
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.FAILED,
                message=f"Plan validation failed: {val_result.error}",
                completed_steps=0,
                total_steps=0,
                failed_step_index=None,
                execution_history=self._execution_history,
            )

        normalized_plan = val_result.normalized_plan
        action_sequence: List[Dict[str, Any]] = normalized_plan["action_sequence"]
        total_steps = len(action_sequence)

        # 2. Check Execution Backend Connectivity
        if not self.is_backend_connected():
            self._status = ExecutionStatus.BACKEND_UNAVAILABLE
            err = (
                "Execution backend is not connected. Action execution is unavailable. "
                "Actions will not be executed or simulated without a connected backend."
            )
            self._log_error(err)
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.BACKEND_UNAVAILABLE,
                message=err,
                completed_steps=0,
                total_steps=total_steps,
                failed_step_index=None,
                execution_history=self._execution_history,
            )

        # 3. Begin Sequential Execution
        self._status = ExecutionStatus.RUNNING
        self._cancel_requested = False
        self._execution_history.clear()
        self._log_info(f"Starting sequential execution of {total_steps} step(s).")

        for idx, action in enumerate(action_sequence):
            self._current_action_index = idx
            self._current_action = action
            intent = action["intent"]

            # Check if cancellation was requested asynchronously
            if self._cancel_requested:
                self._status = ExecutionStatus.CANCELLED
                msg = f"Execution cancelled before step {idx + 1} ({intent})."
                self._log_info(msg)
                return ExecutionResult(
                    success=False,
                    status=ExecutionStatus.CANCELLED,
                    message=msg,
                    completed_steps=idx,
                    total_steps=total_steps,
                    failed_step_index=idx,
                    execution_history=self._execution_history,
                )

            # Handle STOP action
            if intent == "STOP":
                self._log_info(
                    f"Step {idx + 1}/{total_steps}: STOP action encountered. "
                    "Halting sequential execution as requested. "
                    "(Note: Software cancel request, not hardware E-Stop)."
                )
                self._handle_stop()
                self._status = ExecutionStatus.CANCELLED
                self._execution_history.append({
                    "step_index": idx,
                    "action": action,
                    "status": "STOPPED",
                    "message": "Execution halted by STOP action request.",
                })
                return ExecutionResult(
                    success=True,
                    status=ExecutionStatus.CANCELLED,
                    message="Execution halted by STOP action request.",
                    completed_steps=idx + 1,
                    total_steps=total_steps,
                    failed_step_index=None,
                    execution_history=self._execution_history,
                )

            # Dispatch motion or navigation action
            self._log_info(f"Step {idx + 1}/{total_steps}: Executing action {action}...")
            step_success = self._dispatch_action(action)

            if not step_success:
                self._status = ExecutionStatus.FAILED
                err = f"Step {idx + 1}/{total_steps} ({intent}) failed. Halting sequence execution."
                self._log_error(err)
                self._execution_history.append({
                    "step_index": idx,
                    "action": action,
                    "status": "FAILED",
                    "message": err,
                })
                return ExecutionResult(
                    success=False,
                    status=ExecutionStatus.FAILED,
                    message=err,
                    completed_steps=idx,
                    total_steps=total_steps,
                    failed_step_index=idx,
                    execution_history=self._execution_history,
                )

            # Record successful step
            self._execution_history.append({
                "step_index": idx,
                "action": action,
                "status": "SUCCEEDED",
            })
            self._log_info(f"Step {idx + 1}/{total_steps} ({intent}) completed successfully.")

        # 4. Sequence Completed Successfully
        self._status = ExecutionStatus.SUCCEEDED
        self._current_action_index = -1
        self._current_action = None
        success_msg = f"All {total_steps} action(s) completed successfully."
        self._log_info(success_msg)
        return ExecutionResult(
            success=True,
            status=ExecutionStatus.SUCCEEDED,
            message=success_msg,
            completed_steps=total_steps,
            total_steps=total_steps,
            failed_step_index=None,
            execution_history=self._execution_history,
        )

    def _dispatch_action(self, action: Dict[str, Any]) -> bool:
        """Dispatch a single action to registered handlers or the connected backend."""
        intent = action["intent"]

        # Check registered handler first
        if intent in self._action_handlers:
            try:
                return bool(self._action_handlers[intent](action))
            except Exception as exc:
                self._log_error(f"Handler for '{intent}' raised exception: {exc}")
                return False

        # Check connected backend
        if self._backend is not None:
            try:
                return bool(self._backend.execute_action(action))
            except Exception as exc:
                self._log_error(f"Backend execute_action for '{intent}' raised exception: {exc}")
                return False

        self._log_error(f"No handler or backend available to execute intent '{intent}'.")
        return False

    def _handle_stop(self) -> None:
        """Trigger stop request on stop handler or backend."""
        if self._stop_handler is not None:
            try:
                self._stop_handler()
            except Exception as exc:
                self._log_warn(f"Stop handler raised exception: {exc}")

        if self._backend is not None:
            try:
                self._backend.cancel_or_stop()
            except Exception as exc:
                self._log_warn(f"Backend cancel_or_stop raised exception: {exc}")

    # ------------------------------------------------------------------
    # Asynchronous Stop / Cancel Request
    # ------------------------------------------------------------------
    def cancel(self) -> None:
        """
        Request cancellation of current sequence execution.

        NOTE: This is a software-level cancel request and does NOT provide
        a hardware emergency-stop (E-Stop) function.
        """
        self._cancel_requested = True
        self._log_info("Cancellation requested (software cancel, not hardware E-Stop).")
        self._handle_stop()
        self._status = ExecutionStatus.CANCELLED

    def stop(self) -> None:
        """Alias for cancel()."""
        self.cancel()


# ---------------------------------------------------------------------------
# Standalone Demonstration & Self-Test Suite (8 Verification Tests)
# ---------------------------------------------------------------------------
def run_verification_tests() -> bool:
    """Run the 8 standard verification tests for sequential_executor.py."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("=" * 65)
    print("SequentialActionExecutor: Standalone Verification & Demonstration")
    print("=" * 65)

    executor = SequentialActionExecutor(known_locations=["table", "sofa", "bed", "refrigerator"])

    # Test 1: Valid dictionary action plan
    valid_plan = {
        "intent": "SEQUENTIAL",
        "step_count": 3,
        "action_sequence": [
            {"intent": "MOVE_FORWARD", "distance": 1.5},
            {"intent": "TURN_LEFT", "angle": 90.0},
            {"intent": "MOVE_TO_TARGET", "target": "table"},
        ],
    }
    res1 = executor.validate_plan(valid_plan)
    print(f"\n[Test 1] Valid Plan Validation: Passed={res1.is_valid} (Expected: True)")

    # Test 2: Reject string actions missing parameters (no guessing allowed)
    string_plan_missing_params = {
        "intent": "SEQUENTIAL",
        "step_count": 2,
        "action_sequence": ["MOVE_FORWARD", "TURN_LEFT"],
    }
    res2 = executor.validate_plan(string_plan_missing_params)
    print(
        f"[Test 2] String Missing Params Rejected: Passed={not res2.is_valid} "
        f"(Expected: True, Reason: '{res2.error}')"
    )

    # Test 3: Valid STOP structured JSON action (never a bare string)
    stop_plan = {
        "intent": "SEQUENTIAL",
        "step_count": 1,
        "action_sequence": [{"intent": "STOP"}],
    }
    res3 = executor.validate_plan(stop_plan)
    print(f"[Test 3] STOP Structured JSON Action Validation: Passed={res3.is_valid} (Expected: True)")

    # Test 4: Reject unknown target
    unknown_target_plan = {
        "intent": "SEQUENTIAL",
        "step_count": 1,
        "action_sequence": [{"intent": "MOVE_TO_TARGET", "target": "moon_base"}],
    }
    res4 = executor.validate_plan(unknown_target_plan)
    print(
        f"[Test 4] Unknown Target Rejected: Passed={not res4.is_valid} "
        f"(Expected: True, Reason: '{res4.error}')"
    )

    # Test 5: Reject invalid step_count
    mismatch_plan = {
        "intent": "SEQUENTIAL",
        "step_count": 5,
        "action_sequence": [{"intent": "STOP"}],
    }
    res5 = executor.validate_plan(mismatch_plan)
    print(
        f"[Test 5] step_count Mismatch Rejected: Passed={not res5.is_valid} "
        f"(Expected: True, Reason: '{res5.error}')"
    )

    # Test 6: Attempt execution without backend
    exec_res_no_backend = executor.execute_plan(valid_plan)
    print(
        f"[Test 6] Execution Without Backend: Success={exec_res_no_backend.success}, "
        f"Status={exec_res_no_backend.status.value} (Expected: BACKEND_UNAVAILABLE)"
    )

    # Test 7: Execution with a Mock Backend
    class MockBackend(ActionExecutionBackend):
        def __init__(self, fail_on_step: Optional[int] = None):
            self.actions_executed: List[Dict[str, Any]] = []
            self.fail_on_step = fail_on_step

        def execute_action(self, action: Dict[str, Any]) -> bool:
            step_num = len(self.actions_executed) + 1
            if self.fail_on_step == step_num:
                return False
            self.actions_executed.append(action)
            return True

        def cancel_or_stop(self) -> bool:
            return True

    mock_backend = MockBackend()
    executor.set_backend(mock_backend)
    exec_res_mock = executor.execute_plan(valid_plan)
    print(
        f"[Test 7] Execution With Mock Backend: Success={exec_res_mock.success}, "
        f"Completed={exec_res_mock.completed_steps}/{exec_res_mock.total_steps} (Expected: 3/3)"
    )

    # Test 8: Halting on failure during sequential execution
    fail_backend = MockBackend(fail_on_step=2)
    executor.set_backend(fail_backend)
    exec_res_fail = executor.execute_plan(valid_plan)
    print(
        f"[Test 8] Sequence Halts On Step 2 Failure: Success={exec_res_fail.success}, "
        f"Completed={exec_res_fail.completed_steps}, Status={exec_res_fail.status.value} (Expected: FAILED)"
    )

    print("\n" + "=" * 65)
    print("All verification tests completed successfully.")
    print("=" * 65)

    all_passed = (
        res1.is_valid
        and not res2.is_valid
        and res3.is_valid
        and not res4.is_valid
        and not res5.is_valid
        and not exec_res_no_backend.success
        and exec_res_mock.success
        and not exec_res_fail.success
    )
    return all_passed


def main(args=None):
    """Entry point for sequential executor standalone and ROS 2 console script."""
    cli_args = sys.argv[1:]
    if not cli_args:
        # Run standard 8 self-tests
        success = run_verification_tests()
        sys.exit(0 if success else 1)

    json_str = cli_args[0]
    executor = SequentialActionExecutor()

    # Attempt attaching Ros2ActionExecutionBackend if running in ROS 2
    backend = None
    try:
        import rclpy
        from location_navigation.ros2_backend import Ros2ActionExecutionBackend
        if not rclpy.ok():
            rclpy.init(args=args)
        backend = Ros2ActionExecutionBackend()
        executor.set_backend(backend)
    except Exception as exc:
        logging.getLogger(__name__).warning(f"ROS 2 backend not attached: {exc}")

    try:
        result = executor.execute_plan(json_str)
        print(f"Result: success={result.success}, status={result.status.value}, message={result.message}")
    finally:
        if backend is not None:
            backend.destroy()
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
