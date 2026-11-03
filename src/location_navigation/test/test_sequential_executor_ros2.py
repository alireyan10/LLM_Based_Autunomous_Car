#!/usr/bin/env python3
"""
test_sequential_executor_ros2.py
================================
Comprehensive unit test suite for SequentialActionExecutor and Ros2ActionExecutionBackend.

Covers:
1. Preservation of all plan validation rules (intents, parameters, target names, step_count).
2. Parameter safety (rejection of non-positive, non-finite, missing values; no parameter guessing).
3. ROS 2 / Nav2 execution backend integration using mocks.
4. Correct multi-step execution order.
5. Immediate halt of sequence execution on step failure.
6. Asynchronous cancellation and software STOP handling.
7. Action server unavailability and timeout handling.
8. Backward compatibility for single-target navigation and legacy aliases.
9. Explicit rejection of unsupported actions (MOVE_NEAR_TARGET, MOVE_TOWARD_TARGET).
"""

import json
import math
import os
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch

# Add package source to path for importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from location_navigation.sequential_executor import (
    SequentialActionExecutor,
    ExecutionStatus,
    ActionExecutionBackend,
    load_locations_from_yaml,
)
from location_navigation.ros2_backend import (
    Ros2ActionExecutionBackend,
    LEGACY_TARGET_ALIASES,
    DEFAULT_LINEAR_SPEED,
    DEFAULT_ANGULAR_SPEED,
    MAX_FORWARD_DISTANCE,
    MAX_TURN_ANGLE_DEG,
)
from location_navigation.location_database import LocationDatabase
import location_navigation.llm_to_nav


# ===========================================================================
# 1. Plan Validation & Edge Case Tests
# ===========================================================================
class TestPlanValidation(unittest.TestCase):
    """Test plan validation and parameter checking in SequentialActionExecutor."""

    def setUp(self):
        self.known_locations = ["refrigerator", "bed", "sofa", "table"]
        self.executor = SequentialActionExecutor(known_locations=self.known_locations)

    def test_valid_sequential_plan(self):
        """Test valid 3-step action plan validation."""
        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 3,
            "action_sequence": [
                {"intent": "MOVE_FORWARD", "distance": 1.5},
                {"intent": "TURN_LEFT", "angle": 90.0},
                {"intent": "MOVE_TO_TARGET", "target": "table"},
            ],
        }
        res = self.executor.validate_plan(plan)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.normalized_plan["step_count"], 3)
        self.assertEqual(len(res.normalized_plan["action_sequence"]), 3)

    def test_reject_string_actions_missing_parameters(self):
        """String actions must be rejected; every action must be a structured JSON object."""
        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 2,
            "action_sequence": ["MOVE_FORWARD", "TURN_LEFT"],
        }
        res = self.executor.validate_plan(plan)
        self.assertFalse(res.is_valid)
        self.assertIn("Action must be a structured JSON object, never a bare string", res.error)

    def test_valid_stop_structured_action(self):
        """STOP action as a structured JSON object validates successfully without distance/angle."""
        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 1,
            "action_sequence": [{"intent": "STOP"}],
        }
        res = self.executor.validate_plan(plan)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.normalized_plan["action_sequence"][0]["intent"], "STOP")

    def test_reject_bare_string_stop_action(self):
        """Bare string 'STOP' must be rejected per Requirement 2 (must be structured JSON object)."""
        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 1,
            "action_sequence": ["STOP"],
        }
        res = self.executor.validate_plan(plan)
        self.assertFalse(res.is_valid)
        self.assertIn("Action must be a structured JSON object, never a bare string", res.error)

    def test_reject_unknown_target(self):
        """Target names not in the location database must be strictly rejected."""
        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 1,
            "action_sequence": [{"intent": "MOVE_TO_TARGET", "target": "mars_rover"}],
        }
        res = self.executor.validate_plan(plan)
        self.assertFalse(res.is_valid)
        self.assertIn("mars_rover", res.error)

    def test_reject_step_count_mismatch(self):
        """Declared step_count must match length of action_sequence."""
        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 4,
            "action_sequence": [{"intent": "STOP"}],
        }
        res = self.executor.validate_plan(plan)
        self.assertFalse(res.is_valid)
        self.assertIn("step_count", res.error)

    def test_reject_non_positive_or_non_finite_parameters(self):
        """Negative, zero, or NaN/infinite distance and angle values must be rejected."""
        # Negative distance
        plan_neg_dist = {
            "intent": "SEQUENTIAL",
            "step_count": 1,
            "action_sequence": [{"intent": "MOVE_FORWARD", "distance": -1.0}],
        }
        self.assertFalse(self.executor.validate_plan(plan_neg_dist).is_valid)

        # Zero angle
        plan_zero_ang = {
            "intent": "SEQUENTIAL",
            "step_count": 1,
            "action_sequence": [{"intent": "TURN_LEFT", "angle": 0.0}],
        }
        self.assertFalse(self.executor.validate_plan(plan_zero_ang).is_valid)

        # Infinite distance
        plan_inf_dist = {
            "intent": "SEQUENTIAL",
            "step_count": 1,
            "action_sequence": [{"intent": "MOVE_FORWARD", "distance": float("inf")}],
        }
        self.assertFalse(self.executor.validate_plan(plan_inf_dist).is_valid)

    def test_single_action_normalization(self):
        """Single action dictionary should be normalized to 1-step sequential plan."""
        single_action = {"intent": "MOVE_TO_TARGET", "target": "table"}
        res = self.executor.validate_plan(single_action)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.normalized_plan["intent"], "SEQUENTIAL")
        self.assertEqual(res.normalized_plan["step_count"], 1)
        self.assertEqual(res.normalized_plan["action_sequence"][0]["target"], "table")


# ===========================================================================
# 2. Execution Backend & Sequence Lifecycle Tests (Mocked)
# ===========================================================================
class MockRosGoalHandle:
    def __init__(self, accepted=True, status=4):
        self.accepted = accepted
        self.status = status
        self._cancelled = False

    def get_result_async(self):
        future = MagicMock()
        future.done.return_value = True
        result_response = MagicMock()
        result_response.status = self.status
        future.result.return_value = result_response
        return future

    def cancel_goal_async(self):
        self._cancelled = True
        future = MagicMock()
        future.done.return_value = True
        return future


class MockRosActionClient:
    def __init__(self, is_server_ready=True, goal_accepted=True, result_status=4):
        self._server_ready = is_server_ready
        self._goal_accepted = goal_accepted
        self._result_status = result_status
        self.goals_sent = []

    def wait_for_server(self, timeout_sec=10.0):
        return self._server_ready

    def server_is_ready(self):
        return self._server_ready

    def send_goal_async(self, goal_msg):
        self.goals_sent.append(goal_msg)
        future = MagicMock()
        future.done.return_value = True
        future.result.return_value = MockRosGoalHandle(
            accepted=self._goal_accepted,
            status=self._result_status,
        )
        return future


class TestRos2ExecutionBackendMocked(unittest.TestCase):
    """Unit tests for Ros2ActionExecutionBackend with mocked ROS 2 interfaces."""

    def setUp(self):
        # Mock Location Database
        self.mock_db = MagicMock(spec=LocationDatabase)
        self.mock_db.list_locations.return_value = ["refrigerator", "bed", "sofa", "table"]
        self.mock_db.get.side_effect = lambda target: {
            "refrigerator": {"x": 8.353, "y": -3.164, "yaw": 0.0, "frame_id": "map"},
            "table": {"x": 5.920, "y": 0.612, "yaw": 0.0, "frame_id": "map"},
            "sofa": {"x": 1.748, "y": -3.861, "yaw": 0.0, "frame_id": "map"},
            "bed": {"x": -5.168, "y": -0.263, "yaw": 0.0, "frame_id": "map"},
        }[target]

        # Mock ROS Node
        self.mock_node = MagicMock()
        self.mock_node.get_logger.return_value = MagicMock()

        # Patch rclpy in ros2_backend
        self.rclpy_patcher = patch("location_navigation.ros2_backend.rclpy")
        self.mock_rclpy = self.rclpy_patcher.start()
        self.mock_rclpy.ok.return_value = True

        # Instantiate Backend with Mock Node & DB
        self.backend = Ros2ActionExecutionBackend(
            node=self.mock_node,
            location_database=self.mock_db,
        )

        # Wire mock action clients
        self.backend._nav_client = MockRosActionClient(is_server_ready=True, goal_accepted=True, result_status=4)
        self.backend._spin_client = MockRosActionClient(is_server_ready=True, goal_accepted=True, result_status=4)
        self.backend._drive_client = MockRosActionClient(is_server_ready=True, goal_accepted=True, result_status=4)

    def tearDown(self):
        self.rclpy_patcher.stop()

    def test_execute_move_to_target_success(self):
        """MOVE_TO_TARGET resolves target and sends NavigateToPose goal."""
        action = {"intent": "MOVE_TO_TARGET", "target": "table"}
        success = self.backend.execute_action(action)
        self.assertTrue(success)
        self.assertEqual(len(self.backend._nav_client.goals_sent), 1)
        goal = self.backend._nav_client.goals_sent[0]
        self.assertAlmostEqual(goal.pose.pose.position.x, 5.920)
        self.assertAlmostEqual(goal.pose.pose.position.y, 0.612)

    def test_execute_move_forward_success(self):
        """MOVE_FORWARD prefers DriveOnHeading when available."""
        action = {"intent": "MOVE_FORWARD", "distance": 1.5}
        success = self.backend.execute_action(action)
        self.assertTrue(success)
        self.assertEqual(len(self.backend._drive_client.goals_sent), 1)
        goal = self.backend._drive_client.goals_sent[0]
        self.assertAlmostEqual(goal.target.x, 1.5)
        self.assertAlmostEqual(goal.speed, DEFAULT_LINEAR_SPEED)
        # Verify nav client was not used when DriveOnHeading is available
        self.assertEqual(len(self.backend._nav_client.goals_sent), 0)

    def test_execute_move_forward_fallback_to_map_frame_navigate_to_pose(self):
        """When DriveOnHeading is unavailable, MOVE_FORWARD falls back to map-frame NavigateToPose using TF."""
        # DriveOnHeading is unavailable
        self.backend._drive_client = MockRosActionClient(is_server_ready=False)

        # Mock TF current pose lookup: robot at (2.0, 3.0) facing east (yaw=0.0)
        self.backend.get_current_robot_pose_in_map = MagicMock(return_value=(2.0, 3.0, 0.0))

        action = {"intent": "MOVE_FORWARD", "distance": 1.5}
        success = self.backend.execute_action(action)
        self.assertTrue(success)

        # Verified dispatched to NavigateToPose with frame_id='map' (NEVER base_link)
        self.assertEqual(len(self.backend._nav_client.goals_sent), 1)
        goal = self.backend._nav_client.goals_sent[0]
        self.assertEqual(goal.pose.header.frame_id, "map")
        self.assertAlmostEqual(goal.pose.pose.position.x, 3.5)  # 2.0 + 1.5
        self.assertAlmostEqual(goal.pose.pose.position.y, 3.0)  # 3.0 + 0.0

    def test_execute_move_forward_fallback_fails_safely_when_tf_unavailable(self):
        """When DriveOnHeading is unavailable and TF lookup fails, MOVE_FORWARD aborts safely without guessing."""
        self.backend._drive_client = MockRosActionClient(is_server_ready=False)
        # TF lookup returns None (transform failed)
        self.backend.get_current_robot_pose_in_map = MagicMock(return_value=None)

        action = {"intent": "MOVE_FORWARD", "distance": 1.5}
        success = self.backend.execute_action(action)
        self.assertFalse(success)
        # Ensure no goal was sent
        self.assertEqual(len(self.backend._nav_client.goals_sent), 0)

    def test_execute_turn_left_fallback_to_map_frame_navigate_to_pose(self):
        """When Spin is unavailable, TURN_LEFT falls back to map-frame NavigateToPose using TF."""
        self.backend._spin_client = MockRosActionClient(is_server_ready=False)
        self.backend.get_current_robot_pose_in_map = MagicMock(return_value=(2.0, 3.0, 0.0))

        action = {"intent": "TURN_LEFT", "angle": 90.0}
        success = self.backend.execute_action(action)
        self.assertTrue(success)

        self.assertEqual(len(self.backend._nav_client.goals_sent), 1)
        goal = self.backend._nav_client.goals_sent[0]
        self.assertEqual(goal.pose.header.frame_id, "map")
        self.assertAlmostEqual(goal.pose.pose.position.x, 2.0)
        self.assertAlmostEqual(goal.pose.pose.position.y, 3.0)

    def test_execute_turn_left_success(self):
        """TURN_LEFT sends positive yaw to Spin action server."""
        action = {"intent": "TURN_LEFT", "angle": 90.0}
        success = self.backend.execute_action(action)
        self.assertTrue(success)
        self.assertEqual(len(self.backend._spin_client.goals_sent), 1)
        goal = self.backend._spin_client.goals_sent[0]
        self.assertAlmostEqual(goal.target_yaw, math.radians(90.0), places=3)

    def test_execute_turn_right_success(self):
        """TURN_RIGHT sends negative yaw to Spin action server."""
        action = {"intent": "TURN_RIGHT", "angle": 45.0}
        success = self.backend.execute_action(action)
        self.assertTrue(success)
        self.assertEqual(len(self.backend._spin_client.goals_sent), 1)
        goal = self.backend._spin_client.goals_sent[0]
        self.assertAlmostEqual(goal.target_yaw, -math.radians(45.0), places=3)

    def test_reject_unsupported_actions(self):
        """MOVE_NEAR_TARGET and MOVE_TOWARD_TARGET must be explicitly rejected."""
        # MOVE_NEAR_TARGET without standoff offset
        res_near = self.backend.execute_action({"intent": "MOVE_NEAR_TARGET", "target": "sofa"})
        self.assertFalse(res_near)

        # MOVE_TOWARD_TARGET
        res_toward = self.backend.execute_action({"intent": "MOVE_TOWARD_TARGET", "target": "table"})
        self.assertFalse(res_toward)

    def test_server_unavailability_handling(self):
        """When an action server is unavailable, execution must fail safely."""
        self.backend._nav_client = MockRosActionClient(is_server_ready=False)
        success = self.backend.execute_action({"intent": "MOVE_TO_TARGET", "target": "table"})
        self.assertFalse(success)

    def test_goal_rejection_by_action_server(self):
        """When an action server rejects a goal, execution must report failure."""
        self.backend._nav_client = MockRosActionClient(is_server_ready=True, goal_accepted=False)
        success = self.backend.execute_action({"intent": "MOVE_TO_TARGET", "target": "table"})
        self.assertFalse(success)

    def test_action_failure_status(self):
        """When action returns abort or failure status code, execution reports failure."""
        # status 6 = STATUS_ABORTED
        self.backend._nav_client = MockRosActionClient(is_server_ready=True, goal_accepted=True, result_status=6)
        success = self.backend.execute_action({"intent": "MOVE_TO_TARGET", "target": "table"})
        self.assertFalse(success)

    def test_stop_cancels_active_goal(self):
        """STOP action triggers cancel on active goal handle."""
        goal_handle = MockRosGoalHandle()
        self.backend._active_goal_handle = goal_handle

        success = self.backend.cancel_or_stop()
        self.assertTrue(success)
        self.assertTrue(goal_handle._cancelled)
        self.assertIsNone(self.backend._active_goal_handle)


# ===========================================================================
# 3. Full Sequential Execution Integration Tests
# ===========================================================================
class TestSequentialExecutionFlow(unittest.TestCase):
    """Test multi-step execution flow with Ros2ActionExecutionBackend attached."""

    def setUp(self):
        self.mock_db = MagicMock(spec=LocationDatabase)
        self.mock_db.list_locations.return_value = ["table", "sofa", "bed", "refrigerator"]
        self.mock_db.get.return_value = {"x": 5.92, "y": 0.61, "yaw": 0.0, "frame_id": "map"}

        self.mock_node = MagicMock()
        self.mock_node.get_logger.return_value = MagicMock()

        self.rclpy_patcher = patch("location_navigation.ros2_backend.rclpy")
        self.mock_rclpy = self.rclpy_patcher.start()
        self.mock_rclpy.ok.return_value = True

        self.backend = Ros2ActionExecutionBackend(node=self.mock_node, location_database=self.mock_db)
        self.backend._nav_client = MockRosActionClient()
        self.backend._spin_client = MockRosActionClient()
        self.backend._drive_client = MockRosActionClient()

        self.executor = SequentialActionExecutor(
            known_locations=["table", "sofa", "bed", "refrigerator"],
            backend=self.backend,
        )

    def tearDown(self):
        self.rclpy_patcher.stop()

    def test_successful_multi_step_sequence(self):
        """Execute MOVE_FORWARD -> TURN_LEFT -> MOVE_TO_TARGET in strict sequence."""
        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 3,
            "action_sequence": [
                {"intent": "MOVE_FORWARD", "distance": 1.5},
                {"intent": "TURN_LEFT", "angle": 90.0},
                {"intent": "MOVE_TO_TARGET", "target": "table"},
            ],
        }
        res = self.executor.execute_plan(plan)
        self.assertTrue(res.success)
        self.assertEqual(res.status, ExecutionStatus.SUCCEEDED)
        self.assertEqual(res.completed_steps, 3)
        self.assertEqual(len(self.backend._drive_client.goals_sent), 1)
        self.assertEqual(len(self.backend._spin_client.goals_sent), 1)
        self.assertEqual(len(self.backend._nav_client.goals_sent), 1)

    def test_sequence_halts_immediately_on_failure(self):
        """If step 2 fails, execution halts immediately and step 3 is NOT attempted."""
        # Fail spin client (step 2)
        self.backend._spin_client = MockRosActionClient(result_status=6)

        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 3,
            "action_sequence": [
                {"intent": "MOVE_FORWARD", "distance": 1.5},
                {"intent": "TURN_LEFT", "angle": 90.0},
                {"intent": "MOVE_TO_TARGET", "target": "table"},
            ],
        }
        res = self.executor.execute_plan(plan)
        self.assertFalse(res.success)
        self.assertEqual(res.status, ExecutionStatus.FAILED)
        self.assertEqual(res.completed_steps, 1)  # Only Step 1 completed
        self.assertEqual(res.failed_step_index, 1)  # Step 2 failed (0-indexed 1)

        # Confirm step 3 (MOVE_TO_TARGET) was never dispatched
        self.assertEqual(len(self.backend._nav_client.goals_sent), 0)

    def test_stop_action_in_sequence_halts_cleanly(self):
        """STOP action halts remaining sequence actions cleanly."""
        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 3,
            "action_sequence": [
                {"intent": "MOVE_FORWARD", "distance": 1.0},
                {"intent": "STOP"},
                {"intent": "MOVE_TO_TARGET", "target": "table"},
            ],
        }
        res = self.executor.execute_plan(plan)
        self.assertTrue(res.success)
        self.assertEqual(res.status, ExecutionStatus.CANCELLED)
        self.assertEqual(res.completed_steps, 2)
        # Step 3 must not be dispatched
        self.assertEqual(len(self.backend._nav_client.goals_sent), 0)


# ===========================================================================
# 4. Backward Compatibility & Legacy Alias Tests
# ===========================================================================
class TestBackwardCompatibility(unittest.TestCase):
    """Test legacy single-target navigation and alias resolution."""

    def test_legacy_target_aliases_map_to_current_locations(self):
        """Legacy room names must map to the 4 targets in locations.yaml."""
        self.assertEqual(LEGACY_TARGET_ALIASES["kitchen"], "refrigerator")
        self.assertEqual(LEGACY_TARGET_ALIASES["bedroom"], "bed")
        self.assertEqual(LEGACY_TARGET_ALIASES["tv"], "sofa")
        self.assertEqual(LEGACY_TARGET_ALIASES["dining_table"], "table")

    def test_executor_recognizes_legacy_aliases(self):
        """Executor is_known_location returns True for legacy aliases."""
        executor = SequentialActionExecutor(known_locations=["refrigerator", "bed", "sofa", "table"])
        self.assertTrue(executor.is_known_location("refrigerator"))
        self.assertTrue(executor.is_known_location("kitchen"))  # Alias
        self.assertTrue(executor.is_known_location("table"))
        self.assertTrue(executor.is_known_location("dining_table"))  # Alias
        self.assertFalse(executor.is_known_location("bathroom"))


# ===========================================================================
# 5. llm_to_nav Entry Point Integration Tests
# ===========================================================================
class TestLlmToNavCLI(unittest.TestCase):
    """Test llm_to_nav main entry point with different JSON payloads."""

    @patch("location_navigation.llm_to_nav.rclpy")
    @patch("location_navigation.llm_to_nav.Ros2ActionExecutionBackend")
    @patch("location_navigation.llm_to_nav.SequentialActionExecutor")
    def test_llm_to_nav_sequential_plan(self, mock_exec_cls, mock_backend_cls, mock_rclpy):
        """Sequential plan invokes SequentialActionExecutor and exits with code 0 on success."""
        from location_navigation.llm_to_nav import main

        mock_executor = MagicMock()
        mock_res = MagicMock()
        mock_res.success = True
        mock_res.completed_steps = 3
        mock_res.total_steps = 3
        mock_executor.execute_plan.return_value = mock_res
        mock_exec_cls.return_value = mock_executor

        plan_str = json.dumps({
            "intent": "SEQUENTIAL",
            "step_count": 3,
            "action_sequence": [
                {"intent": "MOVE_FORWARD", "distance": 1.5},
                {"intent": "TURN_LEFT", "angle": 90.0},
                {"intent": "MOVE_TO_TARGET", "target": "table"},
            ],
        })

        with patch.object(sys, "argv", ["llm_to_nav", plan_str]):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)
            mock_executor.execute_plan.assert_called_once()

    @patch("location_navigation.llm_to_nav.rclpy")
    @patch("location_navigation.llm_to_nav.NavigationClient")
    def test_llm_to_nav_single_target(self, mock_nav_cls, mock_rclpy):
        """Single MOVE_TO_TARGET uses NavigationClient and exits 0 on success."""
        from location_navigation.llm_to_nav import main

        mock_nav = MagicMock()
        mock_nav.wait_for_server.return_value = True
        mock_nav.navigate_to_pose.return_value = True
        mock_nav_cls.return_value = mock_nav

        single_str = json.dumps({"intent": "MOVE_TO_TARGET", "target": "table"})

        with patch.object(sys, "argv", ["llm_to_nav", single_str]):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)
            mock_nav.navigate_to_pose.assert_called_once()

    @patch("location_navigation.llm_to_nav.rclpy")
    def test_llm_to_nav_unknown_target(self, mock_rclpy):
        """UNKNOWN_TARGET aborts safely with exit code 1."""
        from location_navigation.llm_to_nav import main

        unknown_str = json.dumps({"intent": "UNKNOWN_TARGET", "target": ""})

        with patch.object(sys, "argv", ["llm_to_nav", unknown_str]):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 1)

    @patch("location_navigation.llm_to_nav.rclpy")
    def test_llm_to_nav_invalid_json(self, mock_rclpy):
        """Malformed JSON aborts safely with exit code 1."""
        from location_navigation.llm_to_nav import main

        with patch.object(sys, "argv", ["llm_to_nav", "not a valid json"]):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 1)

    @patch("location_navigation.llm_to_nav.rclpy")
    @patch("location_navigation.llm_to_nav.NavigationClient")
    def test_llm_to_nav_with_ros_args(self, mock_nav_cls, mock_rclpy):
        """JSON plan is correctly extracted even when --ros-args are present."""
        from location_navigation.llm_to_nav import main

        mock_nav = MagicMock()
        mock_nav.wait_for_server.return_value = True
        mock_nav.navigate_to_pose.return_value = True
        mock_nav_cls.return_value = mock_nav

        single_str = json.dumps({"intent": "MOVE_TO_TARGET", "target": "table"})

        # Test with ROS args before JSON
        with patch.object(sys, "argv", ["llm_to_nav", "--ros-args", "-p", "use_sim_time:=true", single_str]):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)
            mock_nav.navigate_to_pose.assert_called_once()

    def test_create_nav_bridge_node_and_annotations(self):
        """Verify create_nav_bridge_node annotations are valid (no NameError for Optional) and node creation succeeds."""
        import inspect
        from location_navigation.llm_to_nav import create_nav_bridge_node

        # Inspect annotations to verify Optional[Node] does not fail with NameError
        annotations = inspect.get_annotations(create_nav_bridge_node)
        self.assertIn("return", annotations)

        # Calling without rclpy returns None safely
        with patch("location_navigation.llm_to_nav.rclpy", None):
            self.assertIsNone(create_nav_bridge_node())

        # Calling with mock rclpy and Node
        mock_node_cls = MagicMock()
        mock_rclpy = MagicMock()
        with patch("location_navigation.llm_to_nav.rclpy", mock_rclpy), \
             patch("location_navigation.llm_to_nav.Node", mock_node_cls):
            node = create_nav_bridge_node("test_bridge")
            mock_node_cls.assert_called()

    def test_extract_plan_argument_scenarios(self):
        """Verify extract_plan_argument correctly strips ROS args and identifies JSON."""
        from location_navigation.llm_to_nav import extract_plan_argument

        # Plain JSON
        self.assertEqual(
            extract_plan_argument(["llm_to_nav", '{"intent":"MOVE_TO_TARGET"}']),
            '{"intent":"MOVE_TO_TARGET"}'
        )
        # With --ros-args preceding JSON
        self.assertEqual(
            extract_plan_argument(["llm_to_nav", "--ros-args", "-p", "use_sim_time:=true", '{"intent":"MOVE_TO_TARGET"}']),
            '{"intent":"MOVE_TO_TARGET"}'
        )
        # Empty args
        self.assertIsNone(extract_plan_argument(["llm_to_nav"]))
        # Non-JSON string
        self.assertEqual(
            extract_plan_argument(["llm_to_nav", "plain_string"]),
            "plain_string"
        )


# ===========================================================================
# 6. Targeted Fixes Tests (Node Spinning, Sim Time, Server Waiting, TF Resilience)
# ===========================================================================
class TestRos2ExecutionFixes(unittest.TestCase):
    """
    Tests for the confirmed issues and fixes:
    1. Background executor spinning lifecycle and thread cleanup.
    2. Simulation time detection and propagation.
    3. Action server waiting timeout configurability (removing hardcoded 2.0s).
    4. TF lookup polling slice resilience, error handling, and cancellation.
    5. Strict sequential execution ordering and fail-safe halting.
    """

    def setUp(self):
        self.mock_db = LocationDatabase.__new__(LocationDatabase)
        self.mock_db._locations = {
            "table": {"x": 1.0, "y": 2.0, "yaw": 0.0, "frame_id": "map"},
            "refrigerator": {"x": 3.0, "y": 4.0, "yaw": 1.57, "frame_id": "map"},
        }

    def test_background_executor_spinning_lifecycle(self):
        """Verify background executor starts when node has no executor, and cleanly stops on destroy()."""
        from location_navigation.ros2_backend import Ros2ActionExecutionBackend

        mock_node = MagicMock()
        mock_node.executor = None
        mock_node.get_logger.return_value = MagicMock()

        # Mock MultiThreadedExecutor with spin that runs until shutdown
        stop_event = threading.Event()
        with patch("location_navigation.ros2_backend.MultiThreadedExecutor") as mock_exec_cls:
            mock_executor_instance = MagicMock()
            mock_executor_instance.spin.side_effect = lambda: stop_event.wait()
            mock_executor_instance.shutdown.side_effect = lambda: stop_event.set()
            mock_exec_cls.return_value = mock_executor_instance

            backend = Ros2ActionExecutionBackend(
                node=mock_node,
                location_database=self.mock_db,
            )

            # Ensure MultiThreadedExecutor was created and node added
            mock_exec_cls.assert_called_once()
            mock_executor_instance.add_node.assert_called_once_with(mock_node)

            # Ensure background thread is running
            self.assertIsNotNone(backend._executor_thread)
            self.assertTrue(backend._executor_thread.is_alive())

            # Destroy backend and verify executor shutdown and thread joined
            thread = backend._executor_thread
            backend.destroy()

            mock_executor_instance.shutdown.assert_called_once()
            self.assertIsNone(backend._executor)
            self.assertIsNone(backend._executor_thread)
            self.assertFalse(thread.is_alive())

    def test_detect_use_sim_time_from_all_sources(self):
        """Verify detect_use_sim_time works with sys.argv, os.environ, parameters, and /clock topic."""
        from location_navigation.ros2_backend import detect_use_sim_time

        # 1. From sys.argv
        with patch.object(sys, "argv", ["node", "--ros-args", "-p", "use_sim_time:=true"]):
            self.assertTrue(detect_use_sim_time())

        with patch.object(sys, "argv", ["node", "use_sim_time:=1"]):
            self.assertTrue(detect_use_sim_time())

        with patch.object(sys, "argv", ["node", "use_sim_time:=false"]):
            self.assertFalse(detect_use_sim_time())

        # 2. From os.environ
        with patch.object(sys, "argv", ["node"]):
            with patch.dict(os.environ, {"USE_SIM_TIME": "true"}):
                self.assertTrue(detect_use_sim_time())
            with patch.dict(os.environ, {"USE_SIM_TIME": "1"}):
                self.assertTrue(detect_use_sim_time())
            with patch.dict(os.environ, {"USE_SIM_TIME": "false"}):
                self.assertFalse(detect_use_sim_time())

        # 3. From node parameter
        with patch.object(sys, "argv", ["node"]):
            with patch.dict(os.environ, {}, clear=True):
                mock_node = MagicMock()
                mock_param = MagicMock()
                mock_param.value = True
                mock_node.get_parameter.return_value = mock_param
                self.assertTrue(detect_use_sim_time(mock_node))

                # Node parameter is False (default) -> should not return True
                mock_param.value = False
                mock_node.count_publishers.return_value = 0
                mock_node.get_topic_names_and_types.return_value = []
                self.assertFalse(detect_use_sim_time(mock_node))

        # 4. From active /clock publisher
        with patch.object(sys, "argv", ["node"]):
            with patch.dict(os.environ, {}, clear=True):
                mock_node = MagicMock()
                mock_node.get_parameter.return_value = MagicMock(value=False)
                mock_node.count_publishers.side_effect = lambda t: 1 if t == "/clock" else 0
                self.assertTrue(detect_use_sim_time(mock_node))

        # 5. From topic list containing /clock
        with patch.object(sys, "argv", ["node"]):
            with patch.dict(os.environ, {}, clear=True):
                mock_node = MagicMock()
                mock_node.get_parameter.return_value = MagicMock(value=False)
                mock_node.count_publishers.return_value = 0
                mock_node.get_topic_names_and_types.return_value = [("/clock", ["rosgraph_msgs/msg/Clock"])]
                self.assertTrue(detect_use_sim_time(mock_node))

        # 6. Default (none present)
        with patch.object(sys, "argv", ["node"]):
            with patch.dict(os.environ, {}, clear=True):
                mock_node = MagicMock()
                mock_node.get_parameter.return_value = MagicMock(value=False)
                mock_node.count_publishers.return_value = 0
                mock_node.get_topic_names_and_types.return_value = []
                self.assertFalse(detect_use_sim_time(mock_node))

    def test_action_server_timeout_configurable(self):
        """Verify action server discovery timeout is configurable and replaces hardcoded 2.0s."""
        from location_navigation.ros2_backend import Ros2ActionExecutionBackend

        mock_node = MagicMock()
        mock_node.executor = MagicMock()  # avoid starting real executor thread
        backend = Ros2ActionExecutionBackend(
            node=mock_node,
            location_database=self.mock_db,
            server_timeout_sec=12.5,
        )
        self.assertEqual(backend._server_timeout_sec, 12.5)

        mock_client = MagicMock()
        mock_client.wait_for_server.return_value = True
        backend._drive_client = mock_client

        # Calling wait_for_server without explicit timeout uses configured server_timeout_sec
        backend.wait_for_server(backend.DRIVE_ACTION)
        mock_client.wait_for_server.assert_called_with(timeout_sec=12.5)

        # Calling with explicit override respects the override
        backend.wait_for_server(backend.DRIVE_ACTION, timeout_sec=3.0)
        mock_client.wait_for_server.assert_called_with(timeout_sec=3.0)

    def test_tf_pose_lookup_resilience_and_cancellation(self):
        """Verify get_current_robot_pose_in_map resilience on TF errors and immediate abort on cancel."""
        from location_navigation.ros2_backend import Ros2ActionExecutionBackend

        mock_node = MagicMock()
        mock_node.executor = MagicMock()
        backend = Ros2ActionExecutionBackend(
            node=mock_node,
            location_database=self.mock_db,
        )

        mock_buffer = MagicMock()
        backend._tf_buffer = mock_buffer

        # Case A: Successful lookup
        mock_transform = MagicMock()
        mock_transform.transform.translation.x = 2.5
        mock_transform.transform.translation.y = 3.5
        mock_transform.transform.rotation.x = 0.0
        mock_transform.transform.rotation.y = 0.0
        mock_transform.transform.rotation.z = 0.0
        mock_transform.transform.rotation.w = 1.0
        mock_buffer.lookup_transform.return_value = mock_transform

        pose = backend.get_current_robot_pose_in_map(timeout_sec=1.0)
        self.assertIsNotNone(pose)
        self.assertAlmostEqual(pose[0], 2.5)
        self.assertAlmostEqual(pose[1], 3.5)
        self.assertAlmostEqual(pose[2], 0.0)

        # Case B: TF buffer raises exception (e.g. "map frame does not exist")
        # Should catch exception, retry until timeout, and safely return None without crashing
        mock_buffer.lookup_transform.side_effect = RuntimeError("map frame does not exist")
        pose = backend.get_current_robot_pose_in_map(timeout_sec=0.2)
        self.assertIsNone(pose)

        # Case C: Cancellation active
        backend._cancel_requested = True
        pose = backend.get_current_robot_pose_in_map(timeout_sec=5.0)
        self.assertIsNone(pose)

    def test_sequential_step_ordering_and_halt_on_step_failure(self):
        """Verify actions run one at a time and sequence immediately stops if an action fails."""
        from location_navigation.sequential_executor import SequentialActionExecutor, ExecutionStatus

        execution_order = []

        class OrderedMockBackend(ActionExecutionBackend):
            def execute_action(self, action):
                intent = action["intent"]
                execution_order.append(intent)
                if intent == "TURN_LEFT":
                    return False  # Fail on step 2
                return True

            def cancel_or_stop(self):
                return True

        backend = OrderedMockBackend()
        executor = SequentialActionExecutor(
            known_locations=["table", "refrigerator", "bed", "sofa"],
            backend=backend,
        )

        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 3,
            "action_sequence": [
                {"intent": "MOVE_FORWARD", "distance": 1.0},
                {"intent": "TURN_LEFT", "angle": 90.0},
                {"intent": "MOVE_TO_TARGET", "target": "table"},
            ],
        }

        result = executor.execute_plan(plan)

        self.assertFalse(result.success)
        self.assertEqual(result.status, ExecutionStatus.FAILED)
        self.assertEqual(result.completed_steps, 1)
        self.assertEqual(result.failed_step_index, 1)
        # Verify Step 1 executed, Step 2 executed and failed, Step 3 was NEVER executed
        self.assertEqual(execution_order, ["MOVE_FORWARD", "TURN_LEFT"])

    def test_action_clients_initialized_with_reentrant_callback_group(self):
        """Verify action clients are configured with a ReentrantCallbackGroup to prevent deadlock."""
        from location_navigation.ros2_backend import Ros2ActionExecutionBackend

        mock_node = MagicMock()
        mock_node.executor = MagicMock()
        mock_cb_group = MagicMock()
        with patch("location_navigation.ros2_backend.NavigateToPose", MagicMock()):
            with patch("location_navigation.ros2_backend.ActionClient") as mock_client_cls:
                with patch("location_navigation.ros2_backend.ReentrantCallbackGroup", return_value=mock_cb_group):
                    backend = Ros2ActionExecutionBackend(
                        node=mock_node,
                        location_database=self.mock_db,
                    )

                    self.assertIsNotNone(backend._action_cb_group)
                    # Verify each ActionClient was called with callback_group
                    self.assertGreaterEqual(mock_client_cls.call_count, 1)
                    for call_args in mock_client_cls.call_args_list:
                        _, kwargs = call_args
                        self.assertIn("callback_group", kwargs)

    def test_spin_fallback_to_navigate_to_pose_on_execution_failure(self):
        """When Spin server is ready but goal execution fails/times out, TURN_LEFT falls back to NavigateToPose via TF."""
        from location_navigation.ros2_backend import Ros2ActionExecutionBackend

        mock_node = MagicMock()
        mock_node.executor = MagicMock()

        backend = Ros2ActionExecutionBackend(
            node=mock_node,
            location_database=self.mock_db,
        )

        # Spin client reports server is ready, but send_goal fails (times out or rejected)
        mock_spin = MockRosActionClient(is_server_ready=True, goal_accepted=False)
        mock_nav = MockRosActionClient(is_server_ready=True, goal_accepted=True, result_status=4)
        backend._spin_client = mock_spin
        backend._nav_client = mock_nav

        # Mock current pose: robot at (1.0, 2.0, yaw=0.0)
        backend.get_current_robot_pose_in_map = MagicMock(return_value=(1.0, 2.0, 0.0))

        action = {"intent": "TURN_LEFT", "angle": 90.0}
        success = backend.execute_action(action)

        self.assertTrue(success)
        self.assertEqual(len(mock_spin.goals_sent), 1)  # Spin was attempted first
        self.assertEqual(len(mock_nav.goals_sent), 1)   # Fallback NavigateToPose was executed and succeeded
        nav_goal = mock_nav.goals_sent[0]
        self.assertEqual(nav_goal.pose.header.frame_id, "map")
        self.assertAlmostEqual(nav_goal.pose.pose.position.x, 1.0)
        self.assertAlmostEqual(nav_goal.pose.pose.position.y, 2.0)

    def test_drive_fallback_to_navigate_to_pose_on_execution_failure(self):
        """When DriveOnHeading server is ready but goal execution fails/times out, MOVE_FORWARD falls back to NavigateToPose via TF."""
        from location_navigation.ros2_backend import Ros2ActionExecutionBackend

        mock_node = MagicMock()
        mock_node.executor = MagicMock()

        backend = Ros2ActionExecutionBackend(
            node=mock_node,
            location_database=self.mock_db,
        )

        # Drive client reports server is ready, but send_goal fails
        mock_drive = MockRosActionClient(is_server_ready=True, goal_accepted=False)
        mock_nav = MockRosActionClient(is_server_ready=True, goal_accepted=True, result_status=4)
        backend._drive_client = mock_drive
        backend._nav_client = mock_nav

        # Mock current pose: robot at (3.0, 4.0, yaw=0.0)
        backend.get_current_robot_pose_in_map = MagicMock(return_value=(3.0, 4.0, 0.0))

        action = {"intent": "MOVE_FORWARD", "distance": 2.0}
        success = backend.execute_action(action)

        self.assertTrue(success)
        self.assertEqual(len(mock_drive.goals_sent), 1)  # Drive was attempted first
        self.assertEqual(len(mock_nav.goals_sent), 1)    # Fallback NavigateToPose was executed and succeeded
        nav_goal = mock_nav.goals_sent[0]
        self.assertEqual(nav_goal.pose.header.frame_id, "map")
        self.assertAlmostEqual(nav_goal.pose.pose.position.x, 5.0)  # 3.0 + 2.0
        self.assertAlmostEqual(nav_goal.pose.pose.position.y, 4.0)

    def test_normalize_yaw(self):
        """Verify normalize_yaw normalizes angles into [-pi, pi]."""
        from location_navigation.ros2_backend import normalize_yaw

        self.assertAlmostEqual(normalize_yaw(0.0), 0.0)
        self.assertAlmostEqual(normalize_yaw(math.pi), math.pi)
        self.assertAlmostEqual(normalize_yaw(-math.pi), -math.pi)
        self.assertAlmostEqual(normalize_yaw(3.0 * math.pi), math.pi)
        self.assertAlmostEqual(normalize_yaw(-3.0 * math.pi), -math.pi)
        self.assertAlmostEqual(normalize_yaw(2.0 * math.pi), 0.0)
        self.assertAlmostEqual(normalize_yaw(math.radians(270.0)), math.radians(-90.0))

    def test_sim_time_elapsed_calculation(self):
        """Verify _get_elapsed_time uses simulation clock when clock is advancing."""
        import time as time_mod
        from location_navigation.ros2_backend import Ros2ActionExecutionBackend

        mock_node = MagicMock()
        mock_node.executor = MagicMock()

        # Mock clock advancing from 10.0s to 15.0s (5.0s elapsed in sim)
        sim_start = MagicMock()
        sim_start.nanoseconds = int(10.0 * 1e9)
        sim_current = MagicMock()
        sim_current.nanoseconds = int(15.0 * 1e9)
        sim_diff = MagicMock()
        sim_diff.nanoseconds = int(5.0 * 1e9)
        sim_current.__ge__.return_value = True
        sim_current.__sub__.return_value = sim_diff

        mock_node.get_clock.return_value.now.return_value = sim_current

        backend = Ros2ActionExecutionBackend(
            node=mock_node,
            location_database=self.mock_db,
        )

        wall_start = time_mod.time()
        elapsed = backend._get_elapsed_time(wall_start, sim_start)
        self.assertAlmostEqual(elapsed, 5.0)

    def test_sim_time_zero_start_clock_falls_back_to_wall_time(self):
        """Verify _get_elapsed_time does NOT return 535.6s when start_sim has nanoseconds=0."""
        import time as time_mod
        from location_navigation.ros2_backend import Ros2ActionExecutionBackend

        mock_node = MagicMock()
        mock_node.executor = MagicMock()

        # sim_start is at 0 (uninitialized sim clock)
        sim_start = MagicMock()
        sim_start.nanoseconds = 0

        # sim_current jumps to 535.6s when Gazebo /clock is received
        sim_current = MagicMock()
        sim_current.nanoseconds = int(535.6 * 1e9)
        sim_current.__ge__.return_value = True

        mock_node.get_clock.return_value.now.return_value = sim_current

        backend = Ros2ActionExecutionBackend(
            node=mock_node,
            location_database=self.mock_db,
        )

        wall_start = time_mod.time() - 0.05
        elapsed = backend._get_elapsed_time(wall_start, sim_start)
        # Must fall back to wall time (~0.05s) and NOT return 535.6s
        self.assertLess(elapsed, 1.0)
        self.assertAlmostEqual(elapsed, 0.05, delta=0.03)

    def test_sim_time_clock_reset_backward_jump(self):
        """Verify _get_elapsed_time falls back to wall time if sim clock jumps backwards."""
        import time as time_mod
        from location_navigation.ros2_backend import Ros2ActionExecutionBackend

        mock_node = MagicMock()
        mock_node.executor = MagicMock()

        sim_start = MagicMock()
        sim_start.nanoseconds = int(500.0 * 1e9)

        # Simulator reset: clock went back to 10.0s
        sim_current = MagicMock()
        sim_current.nanoseconds = int(10.0 * 1e9)
        sim_current.__ge__.return_value = False

        mock_node.get_clock.return_value.now.return_value = sim_current

        backend = Ros2ActionExecutionBackend(
            node=mock_node,
            location_database=self.mock_db,
        )

        wall_start = time_mod.time() - 0.1
        elapsed = backend._get_elapsed_time(wall_start, sim_start)
        self.assertLess(elapsed, 1.0)
        self.assertAlmostEqual(elapsed, 0.1, delta=0.05)

    def test_spin_future_responsively_handles_clock_jump_from_zero(self):
        """Verify _spin_future_responsively handles clock transition from 0 to 535.6s without false timeout."""
        from location_navigation.ros2_backend import Ros2ActionExecutionBackend

        mock_node = MagicMock()
        mock_node.executor = MagicMock()

        # Step 1: clock is 0 when _spin_future_responsively is entered
        clock_zero = MagicMock()
        clock_zero.nanoseconds = 0

        # Step 2: clock becomes 535.6s on subsequent iterations
        clock_sim = MagicMock()
        clock_sim.nanoseconds = int(535.6 * 1e9)
        diff_mock = MagicMock()
        diff_mock.nanoseconds = int(0.05 * 1e9)
        clock_sim.__ge__.return_value = True
        clock_sim.__sub__.return_value = diff_mock

        clock_states = [clock_zero, clock_sim, clock_sim]
        mock_node.get_clock.return_value.now.side_effect = lambda: clock_states.pop(0) if clock_states else clock_sim

        backend = Ros2ActionExecutionBackend(
            node=mock_node,
            location_database=self.mock_db,
        )

        mock_future = MagicMock()
        # Future completes on 2nd check
        done_states = [False, True]
        mock_future.done.side_effect = lambda: done_states.pop(0) if done_states else True

        completed, timed_out_or_cancelled = backend._spin_future_responsively(
            mock_future, timeout_sec=15.0
        )

        self.assertTrue(completed)
        self.assertFalse(timed_out_or_cancelled)

    def test_send_and_await_goal_succeeds_when_clock_jumps_from_zero(self):
        """Verify _send_and_await_goal does not fail with goal response timeout when clock jumps from 0 to 535.6s."""
        from location_navigation.ros2_backend import Ros2ActionExecutionBackend

        mock_node = MagicMock()
        mock_node.executor = MagicMock()

        # Simulate clock starting at 0 and jumping to 535.6s
        clock_zero = MagicMock()
        clock_zero.nanoseconds = 0
        clock_sim = MagicMock()
        clock_sim.nanoseconds = int(535.6 * 1e9)
        diff_mock = MagicMock()
        diff_mock.nanoseconds = int(0.1 * 1e9)
        clock_sim.__ge__.return_value = True
        clock_sim.__sub__.return_value = diff_mock

        clock_states = [clock_zero, clock_sim]
        mock_node.get_clock.return_value.now.side_effect = lambda: clock_states.pop(0) if clock_states else clock_sim

        backend = Ros2ActionExecutionBackend(
            node=mock_node,
            location_database=self.mock_db,
        )

        # Mock action client and futures
        mock_client = MagicMock()
        send_future = MagicMock()
        send_future.done.return_value = True
        mock_goal_handle = MagicMock()
        mock_goal_handle.accepted = True
        send_future.result.return_value = mock_goal_handle
        mock_client.send_goal_async.return_value = send_future

        result_future = MagicMock()
        result_future.done.return_value = True
        mock_result = MagicMock()
        mock_result.status = 4  # STATUS_SUCCEEDED
        result_future.result.return_value = mock_result
        mock_goal_handle.get_result_async.return_value = result_future

        success = backend._send_and_await_goal(
            client=mock_client,
            goal_msg=MagicMock(),
            action_label="MOVE_NEAR_TARGET(bed)",
            timeout_sec=120.0,
        )

        self.assertTrue(success)
        mock_client.send_goal_async.assert_called_once()
        mock_goal_handle.get_result_async.assert_called_once()

    def test_fallback_timeout_sec_default_and_configurability(self):
        """Verify DEFAULT_FALLBACK_TIMEOUT_SEC is 120.0s and properly configurable."""
        from location_navigation.ros2_backend import (
            DEFAULT_FALLBACK_TIMEOUT_SEC,
            Ros2ActionExecutionBackend,
        )

        self.assertEqual(DEFAULT_FALLBACK_TIMEOUT_SEC, 120.0)

        mock_node = MagicMock()
        mock_node.executor = MagicMock()
        mock_node.has_parameter.return_value = False

        # 1. Default timeout should be 120.0 seconds
        backend_default = Ros2ActionExecutionBackend(
            node=mock_node,
            location_database=self.mock_db,
        )
        self.assertEqual(backend_default.fallback_timeout_sec, 120.0)

        # 2. Configurable via constructor
        backend_custom = Ros2ActionExecutionBackend(
            node=mock_node,
            location_database=self.mock_db,
            fallback_timeout_sec=95.0,
        )
        self.assertEqual(backend_custom.fallback_timeout_sec, 95.0)

        # 3. Configurable via property setter
        backend_custom.fallback_timeout_sec = 60.0
        self.assertEqual(backend_custom.fallback_timeout_sec, 60.0)

        # 4. Configurable via ROS node parameter
        mock_param_node = MagicMock()
        mock_param_node.executor = MagicMock()
        mock_param_node.has_parameter.return_value = True
        mock_param = MagicMock()
        mock_param.value = 150.0
        mock_param_node.get_parameter.return_value = mock_param

        backend_param = Ros2ActionExecutionBackend(
            node=mock_param_node,
            location_database=self.mock_db,
        )
        self.assertEqual(backend_param.fallback_timeout_sec, 150.0)

    def test_move_forward_fallback_uses_120s_timeout(self):
        """Verify MOVE_FORWARD NavigateToPose fallback uses 120.0s timeout instead of 25.0s."""
        from location_navigation.ros2_backend import Ros2ActionExecutionBackend

        mock_node = MagicMock()
        mock_node.executor = MagicMock()
        backend = Ros2ActionExecutionBackend(
            node=mock_node,
            location_database=self.mock_db,
        )

        # DriveOnHeading fails/unavailable
        backend._drive_client = MockRosActionClient(is_server_ready=True, goal_accepted=False)
        backend._nav_client = MockRosActionClient(is_server_ready=True, goal_accepted=True, result_status=4)
        backend.get_current_robot_pose_in_map = MagicMock(return_value=(0.0, 0.0, 0.0))

        calls = []
        original_send = backend._send_and_await_goal

        def spy_send(client, goal_msg, action_label, timeout_sec):
            calls.append((action_label, timeout_sec))
            return original_send(client, goal_msg, action_label, timeout_sec)

        backend._send_and_await_goal = spy_send

        # 2m forward move: previously evaluated to (2.0/0.2)+15 = 25.0s
        success = backend.execute_action({"intent": "MOVE_FORWARD", "distance": 2.0})
        self.assertTrue(success)

        # Two calls: 1st was DriveOnHeading, 2nd was NavigateToPose fallback
        self.assertEqual(len(calls), 2)
        fallback_call = calls[1]
        self.assertIn("map-frame NavigateToPose", fallback_call[0])
        self.assertEqual(fallback_call[1], 120.0)  # Must be 120.0s (not 25.0s)

    def test_turn_fallback_uses_120s_timeout(self):
        """Verify TURN NavigateToPose fallback uses 120.0s timeout."""
        from location_navigation.ros2_backend import Ros2ActionExecutionBackend

        mock_node = MagicMock()
        mock_node.executor = MagicMock()
        backend = Ros2ActionExecutionBackend(
            node=mock_node,
            location_database=self.mock_db,
        )

        # Spin fails/unavailable
        backend._spin_client = MockRosActionClient(is_server_ready=True, goal_accepted=False)
        backend._nav_client = MockRosActionClient(is_server_ready=True, goal_accepted=True, result_status=4)
        backend.get_current_robot_pose_in_map = MagicMock(return_value=(0.0, 0.0, 0.0))

        calls = []
        original_send = backend._send_and_await_goal

        def spy_send(client, goal_msg, action_label, timeout_sec):
            calls.append((action_label, timeout_sec))
            return original_send(client, goal_msg, action_label, timeout_sec)

        backend._send_and_await_goal = spy_send

        # 90 deg turn
        success = backend.execute_action({"intent": "TURN_LEFT", "angle": 90.0})
        self.assertTrue(success)

        # Two calls: 1st was Spin, 2nd was NavigateToPose fallback
        self.assertEqual(len(calls), 2)
        fallback_call = calls[1]
        self.assertIn("map-frame NavigateToPose", fallback_call[0])
        self.assertEqual(fallback_call[1], 120.0)  # Must be 120.0s

    def test_fallback_timeout_triggers_goal_cancellation(self):
        """Verify that when fallback NavigateToPose times out, cancel_goal_async is dispatched."""
        from location_navigation.ros2_backend import Ros2ActionExecutionBackend

        mock_node = MagicMock()
        mock_node.executor = MagicMock()
        backend = Ros2ActionExecutionBackend(
            node=mock_node,
            location_database=self.mock_db,
            fallback_timeout_sec=0.05,  # Short timeout for test
        )

        # Mock an action client where send_goal succeeds, but result never arrives
        mock_nav = MagicMock()
        send_future = MagicMock()
        send_future.done.return_value = True
        mock_goal_handle = MagicMock()
        mock_goal_handle.accepted = True

        result_future = MagicMock()
        result_future.done.return_value = False  # Never completes -> times out

        cancel_future = MagicMock()
        cancel_future.done.return_value = True
        mock_goal_handle.get_result_async.return_value = result_future
        mock_goal_handle.cancel_goal_async.return_value = cancel_future

        send_future.result.return_value = mock_goal_handle
        mock_nav.send_goal_async.return_value = send_future
        mock_nav.wait_for_server.return_value = True

        backend._nav_client = mock_nav

        # Test timeout cancellation via _send_and_await_goal with 0.01s timeout
        success = backend._send_and_await_goal(
            client=mock_nav,
            goal_msg=MagicMock(),
            action_label="NavigateToPose fallback",
            timeout_sec=0.01,
        )
        self.assertFalse(success)
        # Verify cancel_goal_async was called because the action timed out
        mock_goal_handle.cancel_goal_async.assert_called_once()

    def test_llm_to_nav_get_fallback_timeout_sec(self):
        """Verify llm_to_nav get_fallback_timeout_sec resolves default and env var overrides."""
        import os
        from location_navigation.llm_to_nav import get_fallback_timeout_sec

        # Default
        self.assertEqual(get_fallback_timeout_sec(None), 120.0)

        # Environment variable override
        with patch.dict(os.environ, {"NAV_FALLBACK_TIMEOUT_SEC": "99.0"}):
            self.assertEqual(get_fallback_timeout_sec(None), 99.0)

        with patch.dict(os.environ, {"FALLBACK_TIMEOUT_SEC": "45.0"}):
            # Clean NAV_FALLBACK_TIMEOUT_SEC if present
            if "NAV_FALLBACK_TIMEOUT_SEC" in os.environ:
                del os.environ["NAV_FALLBACK_TIMEOUT_SEC"]
            self.assertEqual(get_fallback_timeout_sec(None), 45.0)


# ===========================================================================
# 7. Consistent Sequential Schema, Defaults, Limits & MOVE_BACKWARD Tests
# ===========================================================================
class TestConsistentSequentialSchemaAndExecution(unittest.TestCase):
    """
    Comprehensive tests verifying:
    - Every supported action (MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, MOVE_TO_TARGET, STOP).
    - Omitted parameters (safe defaults applied).
    - Explicit parameters preserved.
    - Invalid values rejected (negative, zero, non-numeric, unknown targets, bare strings).
    - Maximum limits enforced (distance <= 2.0m, angle <= 180.0 deg).
    - Speed configuration (linear default 0.15, max 0.20; angular default 0.30, max 0.50).
    - step_count recalculation from final action_sequence.
    - Mixed action sequence validation and execution.
    - MOVE_BACKWARD execution via DriveOnHeading and TF NavigateToPose fallback.
    - Interactive inference plan normalizer (interactive_inference_fixed.py).
    """

    def setUp(self):
        self.known_locations = ["refrigerator", "bed", "sofa", "table"]
        self.executor = SequentialActionExecutor(known_locations=self.known_locations)

    # ------------------------------------------------------------------
    # A. Every Supported Action
    # ------------------------------------------------------------------
    def test_all_supported_actions_validation(self):
        """Verify every supported action validates into the structured schema."""
        actions = [
            {"intent": "MOVE_FORWARD", "distance": 1.5},
            {"intent": "MOVE_BACKWARD", "distance": 0.8},
            {"intent": "TURN_LEFT", "angle": 45.0},
            {"intent": "TURN_RIGHT", "angle": 60.0},
            {"intent": "MOVE_TO_TARGET", "target": "table"},
            {"intent": "MOVE_TO_TARGET", "target": "kitchen"},  # legacy alias
            {"intent": "STOP"},
        ]
        plan = {
            "intent": "SEQUENTIAL",
            "step_count": len(actions),
            "action_sequence": actions,
        }
        res = self.executor.validate_plan(plan)
        self.assertTrue(res.is_valid, f"Validation failed: {res.error}")
        self.assertEqual(res.normalized_plan["step_count"], 7)
        norm_acts = res.normalized_plan["action_sequence"]
        self.assertEqual(norm_acts[0]["intent"], "MOVE_FORWARD")
        self.assertEqual(norm_acts[1]["intent"], "MOVE_BACKWARD")
        self.assertEqual(norm_acts[2]["intent"], "TURN_LEFT")
        self.assertEqual(norm_acts[3]["intent"], "TURN_RIGHT")
        self.assertEqual(norm_acts[4]["intent"], "MOVE_TO_TARGET")
        self.assertEqual(norm_acts[5]["intent"], "MOVE_TO_TARGET")
        self.assertEqual(norm_acts[6]["intent"], "STOP")

    # ------------------------------------------------------------------
    # B. Omitted Parameters (Defaults Applied)
    # ------------------------------------------------------------------
    def test_omitted_parameters_apply_configured_defaults(self):
        """Omitted parameters must receive the exact required defaults."""
        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 5,
            "action_sequence": [
                {"intent": "MOVE_FORWARD"},      # default 1.0 m
                {"intent": "MOVE_BACKWARD"},     # default 1.0 m
                {"intent": "TURN_LEFT"},         # default 90.0 deg
                {"intent": "TURN_RIGHT"},        # default 90.0 deg
                {"intent": "STOP"},              # requires no distance or angle
            ],
        }
        res = self.executor.validate_plan(plan)
        self.assertTrue(res.is_valid, f"Validation failed: {res.error}")
        norm = res.normalized_plan["action_sequence"]
        self.assertEqual(norm[0], {"intent": "MOVE_FORWARD", "distance": 1.0})
        self.assertEqual(norm[1], {"intent": "MOVE_BACKWARD", "distance": 1.0})
        self.assertEqual(norm[2], {"intent": "TURN_LEFT", "angle": 90.0})
        self.assertEqual(norm[3], {"intent": "TURN_RIGHT", "angle": 90.0})
        self.assertEqual(norm[4], {"intent": "STOP"})

    # ------------------------------------------------------------------
    # C. Explicit Parameters Preserved
    # ------------------------------------------------------------------
    def test_explicit_parameters_preserved_within_limits(self):
        """Explicit parameters within allowed limits must not be overwritten by defaults."""
        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 4,
            "action_sequence": [
                {"intent": "MOVE_FORWARD", "distance": 1.75},
                {"intent": "MOVE_BACKWARD", "distance": 0.5},
                {"intent": "TURN_LEFT", "angle": 30.0},
                {"intent": "TURN_RIGHT", "angle": 120.0},
            ],
        }
        res = self.executor.validate_plan(plan)
        self.assertTrue(res.is_valid)
        norm = res.normalized_plan["action_sequence"]
        self.assertEqual(norm[0]["distance"], 1.75)
        self.assertEqual(norm[1]["distance"], 0.5)
        self.assertEqual(norm[2]["angle"], 30.0)
        self.assertEqual(norm[3]["angle"], 120.0)

    # ------------------------------------------------------------------
    # D. Maximum Limits Enforced
    # ------------------------------------------------------------------
    def test_maximum_limits_enforced(self):
        """Movement distance > 2.0m and turn angle > 180 deg must be rejected."""
        # Exact limit distance (2.0 m) accepted
        res_limit_dist = self.executor.validate_plan({
            "intent": "SEQUENTIAL", "step_count": 1,
            "action_sequence": [{"intent": "MOVE_FORWARD", "distance": 2.0}]
        })
        self.assertTrue(res_limit_dist.is_valid)

        # Exceeding limit distance (2.1 m) rejected
        res_over_dist = self.executor.validate_plan({
            "intent": "SEQUENTIAL", "step_count": 1,
            "action_sequence": [{"intent": "MOVE_FORWARD", "distance": 2.1}]
        })
        self.assertFalse(res_over_dist.is_valid)
        self.assertIn("exceeds maximum safety limit", res_over_dist.error)

        # MOVE_BACKWARD exceeding limit distance (2.5 m) rejected
        res_over_back = self.executor.validate_plan({
            "intent": "SEQUENTIAL", "step_count": 1,
            "action_sequence": [{"intent": "MOVE_BACKWARD", "distance": 2.5}]
        })
        self.assertFalse(res_over_back.is_valid)
        self.assertIn("exceeds maximum safety limit", res_over_back.error)

        # Exact limit turn angle (180.0 deg) accepted
        res_limit_turn = self.executor.validate_plan({
            "intent": "SEQUENTIAL", "step_count": 1,
            "action_sequence": [{"intent": "TURN_LEFT", "angle": 180.0}]
        })
        self.assertTrue(res_limit_turn.is_valid)

        # Exceeding limit turn angle (181.0 deg) rejected
        res_over_turn = self.executor.validate_plan({
            "intent": "SEQUENTIAL", "step_count": 1,
            "action_sequence": [{"intent": "TURN_RIGHT", "angle": 181.0}]
        })
        self.assertFalse(res_over_turn.is_valid)
        self.assertIn("exceeds maximum safety limit", res_over_turn.error)

    # ------------------------------------------------------------------
    # E. Invalid Values Rejected
    # ------------------------------------------------------------------
    def test_invalid_parameter_values_rejected(self):
        """Explicit invalid values must be rejected."""
        # Negative distance
        self.assertFalse(self.executor.validate_plan({
            "intent": "SEQUENTIAL", "step_count": 1,
            "action_sequence": [{"intent": "MOVE_FORWARD", "distance": -0.5}]
        }).is_valid)

        # Zero distance
        self.assertFalse(self.executor.validate_plan({
            "intent": "SEQUENTIAL", "step_count": 1,
            "action_sequence": [{"intent": "MOVE_BACKWARD", "distance": 0.0}]
        }).is_valid)

        # Non-numeric distance
        self.assertFalse(self.executor.validate_plan({
            "intent": "SEQUENTIAL", "step_count": 1,
            "action_sequence": [{"intent": "MOVE_FORWARD", "distance": "far"}]
        }).is_valid)

        # Negative angle
        self.assertFalse(self.executor.validate_plan({
            "intent": "SEQUENTIAL", "step_count": 1,
            "action_sequence": [{"intent": "TURN_LEFT", "angle": -45.0}]
        }).is_valid)

        # Zero angle
        self.assertFalse(self.executor.validate_plan({
            "intent": "SEQUENTIAL", "step_count": 1,
            "action_sequence": [{"intent": "TURN_RIGHT", "angle": 0.0}]
        }).is_valid)

        # Non-numeric angle
        self.assertFalse(self.executor.validate_plan({
            "intent": "SEQUENTIAL", "step_count": 1,
            "action_sequence": [{"intent": "TURN_LEFT", "angle": "left"}]
        }).is_valid)

        # Unknown location target (never invent unknown target names)
        res_unknown = self.executor.validate_plan({
            "intent": "SEQUENTIAL", "step_count": 1,
            "action_sequence": [{"intent": "MOVE_TO_TARGET", "target": "garage"}]
        })
        self.assertFalse(res_unknown.is_valid)
        self.assertIn("garage", res_unknown.error)

        # Missing target
        self.assertFalse(self.executor.validate_plan({
            "intent": "SEQUENTIAL", "step_count": 1,
            "action_sequence": [{"intent": "MOVE_TO_TARGET"}]
        }).is_valid)

        # Bare strings rejected (never a bare string)
        res_bare = self.executor.validate_plan({
            "intent": "SEQUENTIAL", "step_count": 1,
            "action_sequence": ["MOVE_FORWARD"]
        })
        self.assertFalse(res_bare.is_valid)
        self.assertIn("never a bare string", res_bare.error)

        # Unsupported actions rejected
        res_unsup = self.executor.validate_plan({
            "intent": "SEQUENTIAL", "step_count": 1,
            "action_sequence": [{"intent": "FLY_DRONE"}]
        })
        self.assertFalse(res_unsup.is_valid)
        self.assertIn("Unsupported action intent", res_unsup.error)

    # ------------------------------------------------------------------
    # F. Recalculation of step_count
    # ------------------------------------------------------------------
    def test_recalculate_step_count(self):
        """Omitted step_count must be safely recalculated from final action_sequence."""
        plan_no_step_count = {
            "intent": "SEQUENTIAL",
            "action_sequence": [
                {"intent": "MOVE_FORWARD", "distance": 1.0},
                {"intent": "TURN_LEFT", "angle": 90.0},
                {"intent": "STOP"},
            ],
        }
        res = self.executor.validate_plan(plan_no_step_count)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.normalized_plan["step_count"], 3)

    # ------------------------------------------------------------------
    # G. Speed Configuration in Ros2Backend
    # ------------------------------------------------------------------
    def test_ros2_backend_speed_configuration_limits(self):
        """Ros2ActionExecutionBackend defaults to 0.15 m/s and 0.30 rad/s, clamped to max 0.20 and 0.50."""
        mock_node = MagicMock()
        mock_node.executor = MagicMock()
        mock_db = MagicMock(spec=LocationDatabase)

        backend = Ros2ActionExecutionBackend(
            node=mock_node,
            location_database=mock_db,
        )
        self.assertAlmostEqual(backend._linear_speed, 0.15)
        self.assertAlmostEqual(backend._angular_speed, 0.30)
        self.assertAlmostEqual(backend._max_distance, 2.0)

        # Clamping test: requesting excessive speeds is safely clamped to maximum limits
        clamped_backend = Ros2ActionExecutionBackend(
            node=mock_node,
            location_database=mock_db,
            linear_speed=0.50,    # exceeds max 0.20
            angular_speed=1.20,   # exceeds max 0.50
            max_distance=10.0,    # exceeds max 2.0
        )
        self.assertAlmostEqual(clamped_backend._linear_speed, 0.20)
        self.assertAlmostEqual(clamped_backend._angular_speed, 0.50)
        self.assertAlmostEqual(clamped_backend._max_distance, 2.0)

    # ------------------------------------------------------------------
    # H. MOVE_BACKWARD Execution in Ros2Backend
    # ------------------------------------------------------------------
    def test_execute_move_backward_drive_on_heading(self):
        """MOVE_BACKWARD dispatches negative x target to DriveOnHeading action server."""
        mock_node = MagicMock()
        mock_node.executor = MagicMock()
        mock_db = MagicMock(spec=LocationDatabase)

        backend = Ros2ActionExecutionBackend(node=mock_node, location_database=mock_db)
        backend._drive_client = MockRosActionClient(is_server_ready=True, goal_accepted=True, result_status=4)

        action = {"intent": "MOVE_BACKWARD", "distance": 1.2}
        success = backend.execute_action(action)
        self.assertTrue(success)

        self.assertEqual(len(backend._drive_client.goals_sent), 1)
        goal = backend._drive_client.goals_sent[0]
        self.assertAlmostEqual(goal.target.x, -1.2)
        self.assertAlmostEqual(goal.speed, 0.15)

    def test_execute_move_backward_tf_fallback(self):
        """MOVE_BACKWARD falls back to map-frame NavigateToPose when DriveOnHeading is unavailable."""
        mock_node = MagicMock()
        mock_node.executor = MagicMock()
        mock_db = MagicMock(spec=LocationDatabase)

        backend = Ros2ActionExecutionBackend(node=mock_node, location_database=mock_db)
        backend._drive_client = MockRosActionClient(is_server_ready=False)
        backend._nav_client = MockRosActionClient(is_server_ready=True, goal_accepted=True, result_status=4)

        # Current pose: robot at (3.0, 4.0) facing yaw=0.0 (east)
        backend.get_current_robot_pose_in_map = MagicMock(return_value=(3.0, 4.0, 0.0))

        action = {"intent": "MOVE_BACKWARD", "distance": 1.0}
        success = backend.execute_action(action)
        self.assertTrue(success)

        self.assertEqual(len(backend._nav_client.goals_sent), 1)
        goal = backend._nav_client.goals_sent[0]
        self.assertEqual(goal.pose.header.frame_id, "map")
        # Moving backward along yaw=0 means target_x = 3.0 - 1.0 = 2.0
        self.assertAlmostEqual(goal.pose.pose.position.x, 2.0)
        self.assertAlmostEqual(goal.pose.pose.position.y, 4.0)

    # ------------------------------------------------------------------
    # I. Mixed Action Sequence Execution
    # ------------------------------------------------------------------
    def test_mixed_action_sequence_execution(self):
        """Execute a 6-step mixed action sequence with all action types."""
        mock_node = MagicMock()
        mock_node.executor = MagicMock()
        mock_db = MagicMock(spec=LocationDatabase)
        mock_db.list_locations.return_value = ["table", "sofa", "bed", "refrigerator"]
        mock_db.get.return_value = {"x": 5.92, "y": 0.61, "yaw": 0.0, "frame_id": "map"}

        backend = Ros2ActionExecutionBackend(node=mock_node, location_database=mock_db)
        backend._drive_client = MockRosActionClient()
        backend._spin_client = MockRosActionClient()
        backend._nav_client = MockRosActionClient()

        executor = SequentialActionExecutor(
            known_locations=["table", "sofa", "bed", "refrigerator"],
            backend=backend,
        )

        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 6,
            "action_sequence": [
                {"intent": "MOVE_FORWARD", "distance": 1.0},
                {"intent": "TURN_LEFT", "angle": 90.0},
                {"intent": "MOVE_BACKWARD", "distance": 0.5},
                {"intent": "TURN_RIGHT", "angle": 45.0},
                {"intent": "MOVE_TO_TARGET", "target": "table"},
                {"intent": "STOP"},
            ],
        }
        res = executor.execute_plan(plan)
        self.assertTrue(res.success)
        self.assertEqual(res.completed_steps, 6)
        self.assertEqual(len(backend._drive_client.goals_sent), 2)  # forward & backward
        self.assertEqual(len(backend._spin_client.goals_sent), 2)   # turn left & right
        self.assertEqual(len(backend._nav_client.goals_sent), 1)    # move to target

    # ------------------------------------------------------------------
    # J. Primary Inference Normalizer (interactive_inference_fixed.py)
    # ------------------------------------------------------------------
    def test_interactive_inference_normalizer(self):
        """Verify normalize_and_validate_plan in interactive_inference_fixed.py."""
        inf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../llm/QWEN_LINUX_TRANSFER/inference"))
        if inf_dir not in sys.path:
            sys.path.insert(0, inf_dir)
        sys.path.insert(0, os.path.expanduser("~/case_study/llm/QWEN_LINUX_TRANSFER/inference"))
        from interactive_inference_fixed import normalize_and_validate_plan

        # Test valid multi-step raw plan with omitted parameters
        raw = {
            "intent": "SEQUENTIAL",
            "action_sequence": [
                {"intent": "MOVE_FORWARD"},
                {"intent": "TURN_LEFT"},
                {"intent": "MOVE_TO_TARGET", "target": "kitchen"},
                {"intent": "STOP"},
            ]
        }
        ok, norm_plan, err = normalize_and_validate_plan(raw)
        self.assertTrue(ok, f"Error: {err}")
        self.assertEqual(norm_plan["step_count"], 4)
        self.assertEqual(norm_plan["action_sequence"][0]["distance"], 1.0)
        self.assertEqual(norm_plan["action_sequence"][1]["angle"], 90.0)
        self.assertEqual(norm_plan["action_sequence"][2]["target"], "kitchen")
        self.assertEqual(norm_plan["action_sequence"][3]["intent"], "STOP")

        # Test bare strings normalized to structured objects
        bare_raw = {"intent": "SEQUENTIAL", "action_sequence": ["STOP"]}
        ok2, norm2, err2 = normalize_and_validate_plan(bare_raw)
        self.assertTrue(ok2, f"Failed: {err2}")
        self.assertEqual(norm2["step_count"], 1)
        self.assertEqual(norm2["action_sequence"][0], {"intent": "STOP"})

        # Test single action normalization (preserves single-target navigation)
        single_raw = {"intent": "MOVE_TO_TARGET", "target": "table"}
        ok3, norm3, _ = normalize_and_validate_plan(single_raw)
        self.assertTrue(ok3)
        self.assertEqual(norm3["intent"], "SEQUENTIAL")
        self.assertEqual(norm3["step_count"], 1)
        self.assertEqual(norm3["action_sequence"][0]["target"], "table")

    def test_regression_failing_bangla_bare_action_strings(self):
        """Regression test: verify exact failing JSON with bare action strings from Qwen is normalized and executed."""
        inf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../llm/QWEN_LINUX_TRANSFER/inference"))
        if inf_dir not in sys.path:
            sys.path.insert(0, inf_dir)
        sys.path.insert(0, os.path.expanduser("~/case_study/llm/QWEN_LINUX_TRANSFER/inference"))
        from interactive_inference_fixed import normalize_and_validate_plan

        # Exact failing JSON produced by Qwen2.5-3B for Bangla command:
        # "সামনে গিয়ে বামে ঘুরো, তারপর থামো।"
        failing_json = {
            "intent": "SEQUENTIAL",
            "step_count": 3,
            "action_sequence": [
                "MOVE_FORWARD",
                "TURN_LEFT",
                "STOP"
            ]
        }

        # 1. Verify strict validation in sequential_executor.py is PRESERVED (rejects bare strings directly)
        executor_direct = SequentialActionExecutor(known_locations=["refrigerator", "bed", "sofa", "table"])
        res_direct = executor_direct.validate_plan(failing_json)
        self.assertFalse(res_direct.is_valid)
        self.assertIn("Action must be a structured JSON object, never a bare string", res_direct.error)

        # 2. Verify normalize_and_validate_plan normalizes the bare strings into structured schema
        ok, norm_plan, err = normalize_and_validate_plan(failing_json)
        self.assertTrue(ok, f"Normalization failed: {err}")
        self.assertEqual(norm_plan["intent"], "SEQUENTIAL")
        self.assertEqual(norm_plan["step_count"], 3)
        self.assertEqual(len(norm_plan["action_sequence"]), 3)

        # 3. Verify step 1: MOVE_FORWARD with distance 1.0
        self.assertEqual(norm_plan["action_sequence"][0], {"intent": "MOVE_FORWARD", "distance": 1.0})

        # 4. Verify step 2: TURN_LEFT with angle 90.0
        self.assertEqual(norm_plan["action_sequence"][1], {"intent": "TURN_LEFT", "angle": 90.0})

        # 5. Verify step 3: STOP as {"intent": "STOP"} (no distance or angle)
        self.assertEqual(norm_plan["action_sequence"][2], {"intent": "STOP"})

        # 6. Verify sequential_executor accepts and executes the normalized plan
        mock_node = MagicMock()
        mock_node.executor = MagicMock()
        mock_db = MagicMock(spec=LocationDatabase)
        mock_db.list_locations.return_value = ["table", "sofa", "bed", "refrigerator"]

        backend = Ros2ActionExecutionBackend(node=mock_node, location_database=mock_db)
        backend._drive_client = MockRosActionClient()
        backend._spin_client = MockRosActionClient()
        backend._nav_client = MockRosActionClient()

        executor = SequentialActionExecutor(
            known_locations=["refrigerator", "bed", "sofa", "table"],
            backend=backend
        )
        val_res = executor.validate_plan(norm_plan)
        self.assertTrue(val_res.is_valid, f"SequentialActionExecutor rejected normalized plan: {val_res.error}")

        exec_res = executor.execute_plan(norm_plan)
        self.assertTrue(exec_res.success)
        self.assertEqual(exec_res.completed_steps, 3)
        self.assertEqual(len(backend._drive_client.goals_sent), 1)
        self.assertEqual(len(backend._spin_client.goals_sent), 1)

    def test_regression_single_target_bed_pipeline(self):
        """Regression Task 6 & 7: Verify pipeline for {'intent': 'MOVE_TO_TARGET', 'target': 'bed'}."""
        inf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../llm/QWEN_LINUX_TRANSFER/inference"))
        if inf_dir not in sys.path:
            sys.path.insert(0, inf_dir)
        sys.path.insert(0, os.path.expanduser("~/case_study/llm/QWEN_LINUX_TRANSFER/inference"))
        from interactive_inference_fixed import normalize_and_validate_plan

        single_target_plan = {"intent": "MOVE_TO_TARGET", "target": "bed"}
        ok, norm_plan, err = normalize_and_validate_plan(single_target_plan)
        self.assertTrue(ok, f"Normalization failed: {err}")
        self.assertEqual(norm_plan["intent"], "SEQUENTIAL")
        self.assertEqual(norm_plan["step_count"], 1)
        self.assertEqual(norm_plan["action_sequence"][0], {"intent": "MOVE_TO_TARGET", "target": "bed"})
        self.assertEqual(norm_plan.get("target"), "bed")

        # Verify ROS 2 dispatch via SequentialActionExecutor and Ros2ActionExecutionBackend
        mock_node = MagicMock()
        mock_node.executor = MagicMock()
        mock_db = MagicMock(spec=LocationDatabase)
        mock_db.list_locations.return_value = ["table", "sofa", "bed", "refrigerator"]
        mock_db.get.return_value = {"x": -5.16869, "y": -0.26320, "yaw": 0.0, "frame_id": "map"}

        backend = Ros2ActionExecutionBackend(node=mock_node, location_database=mock_db)
        backend._nav_client = MockRosActionClient()

        executor = SequentialActionExecutor(
            known_locations=["refrigerator", "bed", "sofa", "table"],
            backend=backend
        )

        val_res = executor.validate_plan(norm_plan)
        self.assertTrue(val_res.is_valid, f"SequentialActionExecutor rejected normalized plan: {val_res.error}")

        exec_res = executor.execute_plan(norm_plan)
        self.assertTrue(exec_res.success)
        self.assertEqual(exec_res.completed_steps, 1)
        self.assertEqual(len(backend._nav_client.goals_sent), 1)
        sent_goal = backend._nav_client.goals_sent[0]
        self.assertAlmostEqual(sent_goal.pose.pose.position.x, -5.16869, places=4)
        self.assertAlmostEqual(sent_goal.pose.pose.position.y, -0.26320, places=4)

    def test_regression_actual_move_near_target_model_output_pipeline(self):
        """
        Regression Tasks 4, 6 & 7: Verify pipeline on actual model output for 'বিছানার কাছে যাও':
        {
          "intent": "MOVE_NEAR_TARGET",
          "target": "bed",
          "speed_category": "NORMAL",
          "linear_speed": 0.15,
          "angular_speed": 0.0,
          "action_sequence": ["MOVE_NEAR_TARGET"],
          "step_count": 1
        }
        """
        inf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../llm/QWEN_LINUX_TRANSFER/inference"))
        if inf_dir not in sys.path:
            sys.path.insert(0, inf_dir)
        sys.path.insert(0, os.path.expanduser("~/case_study/llm/QWEN_LINUX_TRANSFER/inference"))
        from interactive_inference_fixed import normalize_and_validate_plan

        actual_model_output = {
            "intent": "MOVE_NEAR_TARGET",
            "target": "bed",
            "speed_category": "NORMAL",
            "linear_speed": 0.15,
            "angular_speed": 0.0,
            "action_sequence": ["MOVE_NEAR_TARGET"],
            "step_count": 1,
        }

        # 1. Backend behavior check (Task 4): backend rejects raw MOVE_NEAR_TARGET action directly
        mock_node = MagicMock()
        mock_node.executor = MagicMock()
        mock_db = MagicMock(spec=LocationDatabase)
        mock_db.list_locations.return_value = ["table", "sofa", "bed", "refrigerator"]
        mock_db.get.return_value = {"x": -5.16869, "y": -0.26320, "yaw": 0.0, "frame_id": "map"}

        backend = Ros2ActionExecutionBackend(node=mock_node, location_database=mock_db)
        backend._nav_client = MockRosActionClient()
        self.assertFalse(backend.execute_action({"intent": "MOVE_NEAR_TARGET", "target": "bed"}))

        # 2. Normalization check: preserves MOVE_NEAR_TARGET and preserves target="bed"
        ok, norm_plan, err = normalize_and_validate_plan(actual_model_output)
        self.assertTrue(ok, f"Normalization failed on actual model output: {err}")
        self.assertEqual(norm_plan["intent"], "SEQUENTIAL")
        self.assertEqual(norm_plan["step_count"], 1)
        self.assertEqual(norm_plan["action_sequence"][0], {"intent": "MOVE_NEAR_TARGET", "target": "bed"})
        self.assertEqual(norm_plan.get("target"), "bed")

        # 3. Execution check: SequentialActionExecutor accepts plan and dispatches safe approach pose to Nav2
        mock_db.get.return_value = {
            "x": -5.16869,
            "y": -0.26320,
            "yaw": 0.0,
            "frame_id": "map",
            "approach_direction": 0.0,
            "standoff_distance": 0.8,
            "approach_x": -4.36869,
            "approach_y": -0.26320,
            "approach_yaw": math.pi,
        }
        executor = SequentialActionExecutor(
            known_locations=["refrigerator", "bed", "sofa", "table"],
            backend=backend
        )

        val_res = executor.validate_plan(norm_plan)
        self.assertTrue(val_res.is_valid, f"SequentialActionExecutor rejected normalized plan: {val_res.error}")

        exec_res = executor.execute_plan(norm_plan)
        self.assertTrue(exec_res.success)
        self.assertEqual(exec_res.completed_steps, 1)
        self.assertEqual(len(backend._nav_client.goals_sent), 1)
        sent_goal = backend._nav_client.goals_sent[0]
        self.assertAlmostEqual(sent_goal.pose.pose.position.x, -4.36869, places=4)
        self.assertAlmostEqual(sent_goal.pose.pose.position.y, -0.26320, places=4)

    def test_regression_5step_sequential_nav_pipeline(self):
        """
        Regression Tasks 6 & 7: Verify pipeline on 5-step command:
        'সামনে যাও, বামে ঘুরো, সামনে যাও, ডানে ঘুরো, তারপর থামো'
        """
        inf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../llm/QWEN_LINUX_TRANSFER/inference"))
        if inf_dir not in sys.path:
            sys.path.insert(0, inf_dir)
        sys.path.insert(0, os.path.expanduser("~/case_study/llm/QWEN_LINUX_TRANSFER/inference"))
        from interactive_inference_fixed import normalize_and_validate_plan

        failing_5step = {
            "intent": "SEQUENTIAL",
            "step_count": 5,
            "action_sequence": ["MOVE_FORWARD", "TURN_LEFT", "MOVE_FORWARD", "TURN_RIGHT", "STOP"]
        }

        # 1. normalize_and_validate_plan applies safe defaults
        ok, norm_plan, err = normalize_and_validate_plan(failing_5step)
        self.assertTrue(ok, f"Normalization failed: {err}")
        self.assertEqual(norm_plan["step_count"], 5)
        self.assertEqual(len(norm_plan["action_sequence"]), 5)

        # 2. Execution through backend
        mock_node = MagicMock()
        mock_node.executor = MagicMock()
        mock_db = MagicMock(spec=LocationDatabase)
        mock_db.list_locations.return_value = ["table", "sofa", "bed", "refrigerator"]

        backend = Ros2ActionExecutionBackend(node=mock_node, location_database=mock_db)
        backend._drive_client = MockRosActionClient()
        backend._spin_client = MockRosActionClient()
        backend._nav_client = MockRosActionClient()

        executor = SequentialActionExecutor(
            known_locations=["refrigerator", "bed", "sofa", "table"],
            backend=backend
        )

        val_res = executor.validate_plan(norm_plan)
        self.assertTrue(val_res.is_valid, f"SequentialActionExecutor rejected plan: {val_res.error}")

        exec_res = executor.execute_plan(norm_plan)
        self.assertTrue(exec_res.success)
        self.assertEqual(exec_res.completed_steps, 5)
        self.assertEqual(len(backend._drive_client.goals_sent), 2)  # 2 forward moves
        self.assertEqual(len(backend._spin_client.goals_sent), 2)   # 2 turns

    def test_regression_invalid_and_unsupported_intents_pipeline(self):
        """Regression Task 6: Verify pipeline correctly rejects invalid/unsupported intents."""
        inf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../llm/QWEN_LINUX_TRANSFER/inference"))
        if inf_dir not in sys.path:
            sys.path.insert(0, inf_dir)
        sys.path.insert(0, os.path.expanduser("~/case_study/llm/QWEN_LINUX_TRANSFER/inference"))
        from interactive_inference_fixed import normalize_and_validate_plan

        # Top-level unknown intent
        ok1, _, err1 = normalize_and_validate_plan({"intent": "FLY_DRONE"})
        self.assertFalse(ok1)
        self.assertIn("Unsupported", err1)

        # MOVE_NEAR_TARGET without target
        ok2, _, err2 = normalize_and_validate_plan({"intent": "MOVE_NEAR_TARGET"})
        self.assertFalse(ok2)
        self.assertIn("requires a 'target' parameter", err2)

        # MOVE_TO_TARGET with unknown target
        ok3, _, err3 = normalize_and_validate_plan({"intent": "MOVE_TO_TARGET", "target": "garage"})
        self.assertFalse(ok3)
        self.assertIn("unknown", err3.lower())

        # MOVE_NEAR_TARGET with unknown target
        ok4, _, err4 = normalize_and_validate_plan({"intent": "MOVE_NEAR_TARGET", "target": "garage"})
        self.assertFalse(ok4)
        self.assertIn("unknown", err4.lower())

        # MOVE_TOWARD_TARGET without target
        ok5, _, err5 = normalize_and_validate_plan({"intent": "MOVE_TOWARD_TARGET"})
        self.assertFalse(ok5)
        self.assertIn("requires a 'target' parameter", err5)

        # MOVE_TOWARD_TARGET with unknown target
        ok6, _, err6 = normalize_and_validate_plan({"intent": "MOVE_TOWARD_TARGET", "target": "garage"})
        self.assertFalse(ok6)
        self.assertIn("unknown", err6.lower())

    def test_actual_move_toward_target_model_output_pipeline(self):
        """Verify pipeline for MOVE_TOWARD_TARGET model output normalized to MOVE_TO_TARGET."""
        inf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../llm/QWEN_LINUX_TRANSFER/inference"))
        if inf_dir not in sys.path:
            sys.path.insert(0, inf_dir)
        sys.path.insert(0, os.path.expanduser("~/case_study/llm/QWEN_LINUX_TRANSFER/inference"))
        from interactive_inference_fixed import normalize_and_validate_plan

        actual_model_output = {
            "intent": "MOVE_TOWARD_TARGET",
            "target": "table",
            "speed_category": "NORMAL",
            "linear_speed": 0.15,
            "angular_speed": 0.0,
            "action_sequence": ["MOVE_TOWARD_TARGET"],
            "step_count": 1,
        }

        # 1. Backend behavior check: backend rejects raw MOVE_TOWARD_TARGET action directly
        mock_node = MagicMock()
        mock_node.executor = MagicMock()
        mock_db = MagicMock(spec=LocationDatabase)
        mock_db.list_locations.return_value = ["table", "sofa", "bed", "refrigerator"]
        mock_db.get.return_value = {"x": -1.5, "y": 2.0, "yaw": 0.0, "frame_id": "map"}

        backend = Ros2ActionExecutionBackend(node=mock_node, location_database=mock_db)
        backend._nav_client = MockRosActionClient()
        self.assertFalse(backend.execute_action({"intent": "MOVE_TOWARD_TARGET", "target": "table"}))

        # 2. Normalization check: maps MOVE_TOWARD_TARGET to MOVE_TO_TARGET and preserves target="table"
        ok, norm_plan, err = normalize_and_validate_plan(actual_model_output)
        self.assertTrue(ok, f"Normalization failed on MOVE_TOWARD_TARGET: {err}")
        self.assertEqual(norm_plan["intent"], "SEQUENTIAL")
        self.assertEqual(norm_plan["step_count"], 1)
        self.assertEqual(norm_plan["action_sequence"][0], {"intent": "MOVE_TO_TARGET", "target": "table"})
        self.assertEqual(norm_plan.get("target"), "table")

        # 3. Execution check: SequentialActionExecutor accepts normalized plan and dispatches to Nav2
        executor = SequentialActionExecutor(
            known_locations=["refrigerator", "bed", "sofa", "table"],
            backend=backend
        )

        val_res = executor.validate_plan(norm_plan)
        self.assertTrue(val_res.is_valid, f"SequentialActionExecutor rejected normalized plan: {val_res.error}")

        exec_res = executor.execute_plan(norm_plan)
        self.assertTrue(exec_res.success)
        self.assertEqual(exec_res.completed_steps, 1)
        self.assertEqual(len(backend._nav_client.goals_sent), 1)
        sent_goal = backend._nav_client.goals_sent[0]
        self.assertAlmostEqual(sent_goal.pose.pose.position.x, -1.5, places=4)
        self.assertAlmostEqual(sent_goal.pose.pose.position.y, 2.0, places=4)



# ===========================================================================
# 11. MOVE_NEAR_TARGET Safe Target-Approach Tests
# ===========================================================================
class TestMoveNearTargetExecution(unittest.TestCase):
    """
    Comprehensive tests for MOVE_NEAR_TARGET execution logic,
    safe approach pose calculation, configurable standoff, and sequential execution.
    """

    def setUp(self):
        yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config/locations.yaml"))
        self.db = LocationDatabase(yaml_path)

        self.mock_node = MagicMock()
        self.mock_node.get_logger.return_value = MagicMock()
        self.mock_node.executor = MagicMock()

        self.backend = Ros2ActionExecutionBackend(
            node=self.mock_node,
            location_database=self.db,
        )
        self.backend._nav_client = MockRosActionClient(is_server_ready=True, goal_accepted=True, result_status=4)
        self.backend._spin_client = MockRosActionClient(is_server_ready=True, goal_accepted=True, result_status=4)
        self.backend._drive_client = MockRosActionClient(is_server_ready=True, goal_accepted=True, result_status=4)

        self.executor = SequentialActionExecutor(
            known_locations=self.db.list_locations(),
            backend=self.backend,
        )

    def test_reproduction_two_step_sequential_bed_then_table(self):
        """
        Verify reproduction command:
        'রোবট, বিছানার কাছে যাও, তারপর টেবিলের কাছে যাও।'
        Expected plan:
        - Step 1: MOVE_NEAR_TARGET(bed)
        - Step 2: MOVE_NEAR_TARGET(table)
        Both complete in sequential order, reporting success only after both steps complete.
        """
        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 2,
            "action_sequence": [
                {"intent": "MOVE_NEAR_TARGET", "target": "bed"},
                {"intent": "MOVE_NEAR_TARGET", "target": "table"},
            ],
        }

        # 1. Validation
        val_res = self.executor.validate_plan(plan)
        self.assertTrue(val_res.is_valid, f"Validation failed: {val_res.error}")

        # 2. Sequential execution
        exec_res = self.executor.execute_plan(plan)
        self.assertTrue(exec_res.success, f"Execution failed: {exec_res.message}")
        self.assertEqual(exec_res.completed_steps, 2)
        self.assertEqual(exec_res.total_steps, 2)

        # 3. Verify sent goals in order
        self.assertEqual(len(self.backend._nav_client.goals_sent), 2)

        # Step 1: bed approach pose (safe offset from -5.1687, -0.2632)
        goal_1 = self.backend._nav_client.goals_sent[0]
        self.assertAlmostEqual(goal_1.pose.pose.position.x, -4.36869, places=3)
        self.assertAlmostEqual(goal_1.pose.pose.position.y, -0.26320, places=3)
        # Verify robot orientation faces the bed
        self.assertAlmostEqual(goal_1.pose.pose.orientation.w, 0.0, places=3)
        self.assertAlmostEqual(goal_1.pose.pose.orientation.z, 1.0, places=3)

        # Step 2: table approach pose (safe offset from 5.9208, 0.6124)
        goal_2 = self.backend._nav_client.goals_sent[1]
        self.assertAlmostEqual(goal_2.pose.pose.position.x, 5.12084, places=3)
        self.assertAlmostEqual(goal_2.pose.pose.position.y, 0.61241, places=3)
        # Verify robot orientation faces the table (yaw=0.0)
        self.assertAlmostEqual(goal_2.pose.pose.orientation.w, 1.0, places=3)
        self.assertAlmostEqual(goal_2.pose.pose.orientation.z, 0.0, places=3)

    def test_move_near_target_custom_standoff_distance(self):
        """Verify configurable standoff distance parameter in action parameters."""
        action = {"intent": "MOVE_NEAR_TARGET", "target": "bed", "standoff": 1.2}
        success = self.backend.execute_action(action)
        self.assertTrue(success)

        self.assertEqual(len(self.backend._nav_client.goals_sent), 1)
        goal = self.backend._nav_client.goals_sent[0]
        # bed target x is -5.16869, approach_direction is 0.0 (along +X)
        # With standoff 1.2m: -5.16869 + 1.2 = -3.96869
        self.assertAlmostEqual(goal.pose.pose.position.x, -3.96869, places=3)
        self.assertAlmostEqual(goal.pose.pose.position.y, -0.26320, places=3)

    def test_reject_invalid_standoff_parameters(self):
        """Verify non-positive, non-finite, and exceeding limit standoff values are rejected."""
        # Non-positive standoff
        res_neg = self.executor.validate_plan({
            "intent": "SEQUENTIAL",
            "step_count": 1,
            "action_sequence": [{"intent": "MOVE_NEAR_TARGET", "target": "bed", "standoff": -0.5}],
        })
        self.assertFalse(res_neg.is_valid)
        self.assertIn("must be positive", res_neg.error)

        # Zero standoff
        res_zero = self.executor.validate_plan({
            "intent": "SEQUENTIAL",
            "step_count": 1,
            "action_sequence": [{"intent": "MOVE_NEAR_TARGET", "target": "bed", "standoff": 0.0}],
        })
        self.assertFalse(res_zero.is_valid)

        # Standoff exceeding safety limit (2.0m)
        res_large = self.executor.validate_plan({
            "intent": "SEQUENTIAL",
            "step_count": 1,
            "action_sequence": [{"intent": "MOVE_NEAR_TARGET", "target": "bed", "standoff": 3.5}],
        })
        self.assertFalse(res_large.is_valid)
        self.assertIn("exceeds maximum safety limit", res_large.error)

    def test_safe_error_when_no_approach_geometry_available(self):
        """
        Requirement 6: If a safe approach pose cannot be determined,
        return a clear and informative error instead of silently navigating to exact target coordinates.
        """
        mock_db_bare = MagicMock(spec=LocationDatabase)
        mock_db_bare.list_locations.return_value = ["bare_target"]
        mock_db_bare.get.return_value = {"x": 2.0, "y": 3.0, "yaw": 0.0, "frame_id": "map"}
        mock_db_bare.get_approach_pose.return_value = None

        backend = Ros2ActionExecutionBackend(node=self.mock_node, location_database=mock_db_bare)
        backend._nav_client = MockRosActionClient()

        # Must return False and NOT send any navigation goal
        success = backend.execute_action({"intent": "MOVE_NEAR_TARGET", "target": "bare_target"})
        self.assertFalse(success)
        self.assertEqual(len(backend._nav_client.goals_sent), 0)

    def test_step_failure_halts_sequence_immediately(self):
        """When Step 1 fails, Step 2 must never be executed."""
        # Nav client rejects Step 1 goal
        self.backend._nav_client = MockRosActionClient(is_server_ready=True, goal_accepted=False)

        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 2,
            "action_sequence": [
                {"intent": "MOVE_NEAR_TARGET", "target": "bed"},
                {"intent": "MOVE_NEAR_TARGET", "target": "table"},
            ],
        }

        exec_res = self.executor.execute_plan(plan)
        self.assertFalse(exec_res.success)
        self.assertEqual(exec_res.completed_steps, 0)
        self.assertEqual(exec_res.failed_step_index, 0)
        # Only 1 goal attempt occurred; step 2 was halted
        self.assertEqual(len(self.backend._nav_client.goals_sent), 1)

    def test_move_to_target_remains_exact_and_unbroken(self):
        """Strict Restriction: MOVE_TO_TARGET behavior must remain exact and unchanged."""
        action = {"intent": "MOVE_TO_TARGET", "target": "bed"}
        success = self.backend.execute_action(action)
        self.assertTrue(success)

        self.assertEqual(len(self.backend._nav_client.goals_sent), 1)
        goal = self.backend._nav_client.goals_sent[0]
        # MOVE_TO_TARGET navigates to exact coordinates from locations.yaml
        self.assertAlmostEqual(goal.pose.pose.position.x, -5.16869, places=3)
        self.assertAlmostEqual(goal.pose.pose.position.y, -0.26320, places=3)


# ===========================================================================
# Execution Fixes, Aliases & Progress Timeout Tests
# ===========================================================================
class TestExecutionFixesAndAliases(unittest.TestCase):
    """Test ROTATE aliases, LocationDatabase aliases, approach clearance, and progress timeout."""

    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_db.get.side_effect = lambda name: {
            "refrigerator": {"x": 8.353, "y": -3.165, "yaw": 0.0, "frame_id": "map", "approach_direction": math.pi, "standoff_distance": 1.0},
            "bed": {"x": -5.169, "y": -0.263, "yaw": 0.0, "frame_id": "map", "approach_direction": 0.0, "standoff_distance": 0.8},
            "table": {"x": 5.921, "y": 0.612, "yaw": 0.0, "frame_id": "map", "approach_direction": math.pi, "standoff_distance": 0.8},
            "sofa": {"x": 1.749, "y": -3.861, "yaw": 0.0, "frame_id": "map", "approach_direction": math.pi/2, "standoff_distance": 0.8},
        }.get(name)

    def test_rotate_actions_in_sequential_executor(self):
        """SequentialActionExecutor normalizes ROTATE_LEFT and ROTATE_RIGHT to TURN_LEFT and TURN_RIGHT."""
        executor = SequentialActionExecutor(known_locations=["table", "bed"])
        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 2,
            "action_sequence": [
                {"intent": "ROTATE_LEFT", "angle": 90.0},
                {"intent": "ROTATE_RIGHT", "angle": 45.0},
            ]
        }
        res = executor.validate_plan(plan)
        self.assertTrue(res.is_valid, f"Validation failed: {res.error}")
        norm_seq = res.normalized_plan["action_sequence"]
        self.assertEqual(norm_seq[0]["intent"], "TURN_LEFT")
        self.assertEqual(norm_seq[0]["angle"], 90.0)
        self.assertEqual(norm_seq[1]["intent"], "TURN_RIGHT")
        self.assertEqual(norm_seq[1]["angle"], 45.0)

    def test_rotate_actions_in_ros2_backend(self):
        """Ros2ActionExecutionBackend maps ROTATE_LEFT and ROTATE_RIGHT to turn execution."""
        mock_node = MagicMock()
        mock_node.executor = MagicMock()
        backend = Ros2ActionExecutionBackend(
            node=mock_node,
            location_database=self.mock_db,
        )
        mock_spin = MockRosActionClient(is_server_ready=True, goal_accepted=True, result_status=4)
        backend._spin_client = mock_spin

        success_left = backend.execute_action({"intent": "ROTATE_LEFT", "angle": 90.0})
        self.assertTrue(success_left)
        self.assertEqual(len(mock_spin.goals_sent), 1)
        self.assertAlmostEqual(mock_spin.goals_sent[0].target_yaw, math.radians(90.0), places=3)

        success_right = backend.execute_action({"intent": "ROTATE_RIGHT", "angle": 45.0})
        self.assertTrue(success_right)
        self.assertEqual(len(mock_spin.goals_sent), 2)
        self.assertAlmostEqual(mock_spin.goals_sent[1].target_yaw, -math.radians(45.0), places=3)

    def test_location_database_aliases_and_approach(self):
        """LocationDatabase correctly resolves bedroom -> bed and kitchen -> refrigerator."""
        db = LocationDatabase("src/location_navigation/config/locations.yaml")
        # get() resolution
        bed_pose = db.get("bed")
        bedroom_pose = db.get("bedroom")
        self.assertEqual(bedroom_pose["x"], bed_pose["x"])
        self.assertEqual(bedroom_pose["y"], bed_pose["y"])

        fridge_pose = db.get("refrigerator")
        kitchen_pose = db.get("kitchen")
        self.assertEqual(kitchen_pose["x"], fridge_pose["x"])
        self.assertEqual(kitchen_pose["y"], fridge_pose["y"])

        # __contains__() resolution
        self.assertIn("bedroom", db)
        self.assertIn("kitchen", db)
        self.assertNotIn("bathroom", db)

        # Approach pose resolution
        app_bedroom = db.get_approach_pose("bedroom")
        self.assertIsNotNone(app_bedroom)
        self.assertAlmostEqual(app_bedroom["y"], bed_pose["y"], places=3)

        app_kitchen = db.get_approach_pose("kitchen")
        self.assertIsNotNone(app_kitchen)
        # Safe standoff >= 1.0m from refrigerator (8.353 - 1.0 = 7.353)
        self.assertAlmostEqual(app_kitchen["x"], 7.3531, places=3)

    def test_adaptive_standoff_fallback_on_status_6(self):
        """When Nav2 aborts initial approach with status code 6, retry with increased standoff."""
        mock_node = MagicMock()
        mock_node.executor = MagicMock()
        db = LocationDatabase("src/location_navigation/config/locations.yaml")
        backend = Ros2ActionExecutionBackend(
            node=mock_node,
            location_database=db,
            fallback_timeout_sec=5.0,
        )

        mock_nav = MagicMock()
        mock_nav.wait_for_server.return_value = True

        # First goal attempt is accepted, then aborts with status code 6
        goal1_handle = MagicMock()
        goal1_handle.accepted = True
        goal1_result_future = MagicMock()
        goal1_result_future.done.return_value = True
        goal1_resp = MagicMock()
        goal1_resp.status = 6  # STATUS_ABORTED
        goal1_result_future.result.return_value = goal1_resp
        goal1_handle.get_result_async.return_value = goal1_result_future

        # Second goal attempt (fallback standoff) succeeds with status code 4
        goal2_handle = MagicMock()
        goal2_handle.accepted = True
        goal2_result_future = MagicMock()
        goal2_result_future.done.return_value = True
        goal2_resp = MagicMock()
        goal2_resp.status = 4  # STATUS_SUCCEEDED
        goal2_result_future.result.return_value = goal2_resp
        goal2_handle.get_result_async.return_value = goal2_result_future

        send1_future = MagicMock()
        send1_future.done.return_value = True
        send1_future.result.return_value = goal1_handle

        send2_future = MagicMock()
        send2_future.done.return_value = True
        send2_future.result.return_value = goal2_handle

        mock_nav.send_goal_async.side_effect = [send1_future, send2_future]
        backend._nav_client = mock_nav

        # Execute MOVE_NEAR_TARGET for refrigerator
        success = backend.execute_action({"intent": "MOVE_NEAR_TARGET", "target": "refrigerator"})
        self.assertTrue(success)
        self.assertEqual(mock_nav.send_goal_async.call_count, 2)

    def test_progress_monitoring_allows_valid_goal_to_finish(self):
        """Goal making continuous progress does not get cancelled at nominal timeout."""
        mock_node = MagicMock()
        mock_node.executor = MagicMock()
        backend = Ros2ActionExecutionBackend(
            node=mock_node,
            location_database=self.mock_db,
            stuck_timeout_sec=2.0,
            max_timeout_sec=10.0,
        )
        mock_tf = MagicMock()
        backend._tf_buffer = mock_tf

        # Simulate robot advancing 0.1m every lookup
        poses = [
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.2, 0.0, 0.0),
            (0.3, 0.0, 0.0),
        ]
        pose_iter = iter(poses)

        def mock_lookup():
            try:
                return next(pose_iter)
            except StopIteration:
                return (0.4, 0.0, 0.0)

        backend._lookup_robot_pose_quiet = mock_lookup

        # Result future that finishes after a brief time
        res_future = MagicMock()
        call_count = [0]
        def is_done():
            call_count[0] += 1
            return call_count[0] > 4
        res_future.done.side_effect = is_done

        completed, timed_out = backend._spin_future_with_progress(
            res_future,
            action_label="TestProgress",
            timeout_sec=0.01,  # Lower than execution duration, but progress keeps it alive
            stuck_timeout_sec=1.0,
            max_timeout_sec=5.0,
        )
        self.assertTrue(completed)
        self.assertFalse(timed_out)

    def test_stuck_detection_cancels_goal_when_no_progress(self):
        """When robot is stationary for stuck_timeout_sec, goal is cancelled."""
        mock_node = MagicMock()
        mock_node.executor = MagicMock()
        backend = Ros2ActionExecutionBackend(
            node=mock_node,
            location_database=self.mock_db,
            stuck_timeout_sec=0.1,  # Fast stuck timeout
            max_timeout_sec=5.0,
        )
        backend._lookup_robot_pose_quiet = lambda: (1.0, 1.0, 0.0)  # Always same pose

        res_future = MagicMock()
        res_future.done.return_value = False  # Never completes

        completed, timed_out = backend._spin_future_with_progress(
            res_future,
            action_label="TestStuck",
            timeout_sec=0.05,
            stuck_timeout_sec=0.1,
            max_timeout_sec=5.0,
        )
        self.assertFalse(completed)
        self.assertTrue(timed_out)


if __name__ == "__main__":
    unittest.main()



