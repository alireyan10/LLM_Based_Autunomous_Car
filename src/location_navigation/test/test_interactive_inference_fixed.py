#!/usr/bin/env python3
"""
test_interactive_inference_fixed.py
===================================
Regression and unit test suite for interactive_inference_fixed.py.

Covers:
1. Stopping Criteria (JsonStoppingCriteria):
   - Dynamic halt when the first JSON object completes.
   - Non-premature halt on inner braces in structured action sequences.
   - Proper stopping on bare-string sequences and single-action objects.
   - Proper return types.
2. Output Parsing (parse_llm_output):
   - Extraction of single valid JSON objects.
   - Markdown code-fence stripping (```json ... ```).
   - Resilience against repeated JSON fragments / extra trailing data.
   - Proper rejection of truncated, malformed, or non-JSON inputs.
3. Plan Normalization & Validation (normalize_and_validate_plan):
   - Preservation of original bare-string action_sequence format:
     ["MOVE_FORWARD", "TURN_LEFT", "STOP"].
   - Application of structured action defaults:
     MOVE_FORWARD distance=1.0
     MOVE_BACKWARD distance=1.0
     TURN_LEFT angle=90.0
     TURN_RIGHT angle=90.0
     STOP with no parameters.
   - Safety limits: distance <= 2.0m, turn angle <= 180.0 deg.
   - Target resolution for known locations and legacy aliases.
   - Support for single-target commands and single actions.
   - Exact recalculation of step_count.
4. End-to-End Simulation of Problem Command:
   - "সামনে যাও, বামে ঘুরো, তারপর থামো"
"""

import json
import math
import os
import sys
import unittest
from unittest.mock import MagicMock

# Add inference and package directories to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
INFERENCE_DIR = os.path.join(BASE_DIR, "llm", "QWEN_LINUX_TRANSFER", "inference")
sys.path.insert(0, INFERENCE_DIR)

from interactive_inference_fixed import (
    DEFAULT_SYSTEM_PROMPT,
    QWEN_CHAT_TEMPLATE,
    configure_chat_template,
    format_prompt,
    parse_llm_output,
    JsonStoppingCriteria,
    normalize_and_validate_plan,
    extract_targets_from_command,
    extract_distances_from_command,
    extract_angles_from_command,
    parse_bangla_or_english_number,
    ACTION_ALIASES,
    ALLOWED_ACTIONS,
)


class MockTokenizer:
    """Mock tokenizer for testing JsonStoppingCriteria."""
    def __init__(self, token_to_text_map=None):
        self.token_to_text_map = token_to_text_map or {}
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = 151643
        self.unk_token_id = 151642

    def decode(self, token_ids, skip_special_tokens=True):
        # If token_ids is a list or tensor-like, look up each or join
        chars = []
        for tid in token_ids:
            if hasattr(tid, "item"):
                tid = tid.item()
            chars.append(self.token_to_text_map.get(tid, str(tid)))
        return "".join(chars)

    def convert_tokens_to_ids(self, token_str):
        if token_str == "<|im_end|>":
            return 151645
        if token_str == "<|endoftext|>":
            return 151643
        return self.unk_token_id


# ===========================================================================
# 1. Tests for JsonStoppingCriteria
# ===========================================================================
class TestJsonStoppingCriteria(unittest.TestCase):
    """Test stopping criteria behavior during generation."""

    def test_stop_on_first_complete_bare_string_json(self):
        """Stopping criteria must trigger when the outer closing brace of a bare-string JSON completes."""
        json_str = '{"intent": "SEQUENTIAL", "step_count": 3, "action_sequence": ["MOVE_FORWARD", "TURN_LEFT", "STOP"]}'
        # Map each character to a token ID
        token_map = {idx: ch for idx, ch in enumerate(json_str)}
        tokenizer = MockTokenizer(token_map)
        
        prompt_length = 10
        criteria = JsonStoppingCriteria(tokenizer, prompt_length)
        
        # Incomplete stream (before closing brace)
        incomplete_ids = [0] * prompt_length + list(range(len(json_str) - 1))
        res = criteria([incomplete_ids], None)
        self.assertFalse(bool(res if not hasattr(res, "item") else res.item()))

        # Complete stream (at closing brace)
        complete_ids = [0] * prompt_length + list(range(len(json_str)))
        res = criteria([complete_ids], None)
        self.assertTrue(bool(res if not hasattr(res, "item") else res.item()))

    def test_do_not_stop_prematurely_on_inner_action_braces(self):
        """Stopping criteria must NOT stop on intermediate closing braces inside action_sequence."""
        # Simulated sequence with inner braces
        full_json = '{"intent": "SEQUENTIAL", "step_count": 2, "action_sequence": [{"intent": "MOVE_FORWARD", "distance": 1.0}, {"intent": "STOP"}]}'
        token_map = {idx: ch for idx, ch in enumerate(full_json)}
        tokenizer = MockTokenizer(token_map)
        
        prompt_length = 5
        criteria = JsonStoppingCriteria(tokenizer, prompt_length)

        # Slice right after the first inner action: {"intent": "MOVE_FORWARD", "distance": 1.0}
        first_inner_end = full_json.find('}') + 1
        inner_slice_ids = [0] * prompt_length + list(range(first_inner_end))
        res = criteria([inner_slice_ids], None)
        # Must NOT stop because outer object and array are still unclosed
        self.assertFalse(bool(res if not hasattr(res, "item") else res.item()))

        # At the very end of the full JSON
        complete_ids = [0] * prompt_length + list(range(len(full_json)))
        res = criteria([complete_ids], None)
        self.assertTrue(bool(res if not hasattr(res, "item") else res.item()))

    def test_stop_on_single_action_json(self):
        """Stopping criteria must stop on single-action JSON object."""
        single_json = '{"intent": "MOVE_TO_TARGET", "target": "kitchen"}'
        token_map = {idx: ch for idx, ch in enumerate(single_json)}
        tokenizer = MockTokenizer(token_map)
        
        prompt_length = 4
        criteria = JsonStoppingCriteria(tokenizer, prompt_length)
        
        complete_ids = [0] * prompt_length + list(range(len(single_json)))
        res = criteria([complete_ids], None)
        self.assertTrue(bool(res if not hasattr(res, "item") else res.item()))


# ===========================================================================
# 2. Tests for parse_llm_output (Regression & Edge Cases)
# ===========================================================================
class TestParseLlmOutput(unittest.TestCase):
    """Test parse_llm_output on clean, repeated fragments, and malformed outputs."""

    def test_clean_json_parsing(self):
        """Standard clean JSON string must parse correctly."""
        raw = '{"intent": "SEQUENTIAL", "step_count": 3, "action_sequence": ["MOVE_FORWARD", "TURN_LEFT", "STOP"]}'
        data, ok = parse_llm_output(raw)
        self.assertTrue(ok)
        self.assertEqual(data["intent"], "SEQUENTIAL")
        self.assertEqual(len(data["action_sequence"]), 3)

    def test_markdown_code_fence_parsing(self):
        """JSON enclosed in markdown ```json ... ``` blocks must parse correctly."""
        raw = '```json\n{"intent": "SEQUENTIAL", "step_count": 1, "action_sequence": ["STOP"]}\n```'
        data, ok = parse_llm_output(raw)
        self.assertTrue(ok)
        self.assertEqual(data["intent"], "SEQUENTIAL")
        self.assertEqual(data["action_sequence"], ["STOP"])

    def test_regression_repeated_json_fragments(self):
        """
        Regression: When the model outputs one valid JSON followed by repeated intent fragments,
        parse_llm_output must cleanly extract the first valid JSON object rather than failing.
        """
        raw = (
            '{"intent": "SEQUENTIAL", "step_count": 3, "action_sequence": ["MOVE_FORWARD", "TURN_LEFT", "STOP"]}\n'
            '{"intent": "MOVE_FORWARD"}\n'
            '{"intent": "TURN_LEFT"}\n'
            '{"intent": "STOP"}'
        )
        data, ok = parse_llm_output(raw)
        self.assertTrue(ok)
        self.assertEqual(data["intent"], "SEQUENTIAL")
        self.assertEqual(data["step_count"], 3)
        self.assertEqual(data["action_sequence"], ["MOVE_FORWARD", "TURN_LEFT", "STOP"])

    def test_regression_repeated_structured_fragments(self):
        """Regression: Structured sequential plan followed by repeated single structured actions."""
        raw = (
            '{"intent": "SEQUENTIAL", "step_count": 2, "action_sequence": [{"intent": "MOVE_FORWARD", "distance": 1.0}, {"intent": "STOP"}]}\n'
            '{"intent": "MOVE_FORWARD", "distance": 1.0}\n'
            '{"intent": "STOP"}'
        )
        data, ok = parse_llm_output(raw)
        self.assertTrue(ok)
        self.assertEqual(data["intent"], "SEQUENTIAL")
        self.assertEqual(len(data["action_sequence"]), 2)

    def test_regression_consecutive_json_objects_without_newline(self):
        """Multiple JSON objects concatenated directly without whitespace."""
        raw = '{"intent": "MOVE_TO_TARGET", "target": "bed"}{"intent": "STOP"}'
        data, ok = parse_llm_output(raw)
        self.assertTrue(ok)
        self.assertEqual(data["intent"], "MOVE_TO_TARGET")
        self.assertEqual(data["target"], "bed")

    def test_malformed_unclosed_json_rejected(self):
        """Incomplete or unclosed JSON must return None, False."""
        raw = '{"intent": "SEQUENTIAL", "step_count": 3, "action_sequence": ['
        data, ok = parse_llm_output(raw)
        self.assertFalse(ok)
        self.assertIsNone(data)

    def test_malformed_syntax_error_rejected(self):
        """Invalid JSON syntax must return None, False."""
        raw = '{"intent": "SEQUENTIAL", "action_sequence": ["MOVE_FORWARD",],}'
        data, ok = parse_llm_output(raw)
        self.assertFalse(ok)
        self.assertIsNone(data)

    def test_garbage_text_rejected(self):
        """Non-JSON text must return None, False."""
        raw = 'I am a robot navigation assistant, please give a valid command.'
        data, ok = parse_llm_output(raw)
        self.assertFalse(ok)
        self.assertIsNone(data)

    def test_empty_string_rejected(self):
        """Empty or whitespace-only string must return None, False."""
        data, ok = parse_llm_output("   \n\t  ")
        self.assertFalse(ok)
        self.assertIsNone(data)


# ===========================================================================
# 3. Tests for normalize_and_validate_plan
# ===========================================================================
class TestNormalizeAndValidatePlan(unittest.TestCase):
    """Test plan normalization, defaults, limits, and bare-string support."""

    def test_original_bare_string_action_sequence(self):
        """
        Requirement 8: The original bare-string action_sequence format must normalize
        into structured actions with appropriate defaults.
        """
        bare_plan = {
            "intent": "SEQUENTIAL",
            "step_count": 3,
            "action_sequence": ["MOVE_FORWARD", "TURN_LEFT", "STOP"],
        }
        ok, norm, err = normalize_and_validate_plan(bare_plan)
        self.assertTrue(ok, f"Validation failed: {err}")
        self.assertEqual(norm["intent"], "SEQUENTIAL")
        self.assertEqual(norm["step_count"], 3)
        self.assertEqual(norm["action_sequence"], [
            {"intent": "MOVE_FORWARD", "distance": 1.0},
            {"intent": "TURN_LEFT", "angle": 90.0},
            {"intent": "STOP"},
        ])

    def test_structured_action_defaults(self):
        """
        Requirement 5: Preserves defaults only for omitted parameters:
        MOVE_FORWARD distance=1.0
        MOVE_BACKWARD distance=1.0
        TURN_LEFT angle=90.0
        TURN_RIGHT angle=90.0
        STOP with no parameters.
        """
        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 5,
            "action_sequence": [
                {"intent": "MOVE_FORWARD"},
                {"intent": "MOVE_BACKWARD"},
                {"intent": "TURN_LEFT"},
                {"intent": "TURN_RIGHT"},
                {"intent": "STOP"},
            ],
        }
        ok, norm, err = normalize_and_validate_plan(plan)
        self.assertTrue(ok, f"Validation failed: {err}")
        self.assertEqual(norm["action_sequence"][0], {"intent": "MOVE_FORWARD", "distance": 1.0})
        self.assertEqual(norm["action_sequence"][1], {"intent": "MOVE_BACKWARD", "distance": 1.0})
        self.assertEqual(norm["action_sequence"][2], {"intent": "TURN_LEFT", "angle": 90.0})
        self.assertEqual(norm["action_sequence"][3], {"intent": "TURN_RIGHT", "angle": 90.0})
        self.assertEqual(norm["action_sequence"][4], {"intent": "STOP"})
        self.assertNotIn("distance", norm["action_sequence"][4])
        self.assertNotIn("angle", norm["action_sequence"][4])

    def test_explicit_valid_parameters_preserved(self):
        """Explicitly provided non-default valid parameters must be preserved."""
        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 2,
            "action_sequence": [
                {"intent": "MOVE_FORWARD", "distance": 1.5},
                {"intent": "TURN_RIGHT", "angle": 45.0},
            ],
        }
        ok, norm, err = normalize_and_validate_plan(plan)
        self.assertTrue(ok, f"Validation failed: {err}")
        self.assertEqual(norm["action_sequence"][0]["distance"], 1.5)
        self.assertEqual(norm["action_sequence"][1]["angle"], 45.0)

    def test_safety_limits_exceeded(self):
        """Distances > 2.0m and angles > 180 deg must be rejected."""
        # Forward > 2.0m
        p1 = {"intent": "SEQUENTIAL", "action_sequence": [{"intent": "MOVE_FORWARD", "distance": 2.1}]}
        ok1, _, err1 = normalize_and_validate_plan(p1)
        self.assertFalse(ok1)
        self.assertIn("exceeds maximum safety limit 2.0m", err1)

        # Backward > 2.0m
        p2 = {"intent": "SEQUENTIAL", "action_sequence": [{"intent": "MOVE_BACKWARD", "distance": 2.5}]}
        ok2, _, err2 = normalize_and_validate_plan(p2)
        self.assertFalse(ok2)
        self.assertIn("exceeds maximum safety limit 2.0m", err2)

        # Turn > 180 deg
        p3 = {"intent": "SEQUENTIAL", "action_sequence": [{"intent": "TURN_LEFT", "angle": 181.0}]}
        ok3, _, err3 = normalize_and_validate_plan(p3)
        self.assertFalse(ok3)
        self.assertIn("exceeds maximum safety limit 180.0 deg", err3)

    def test_invalid_parameter_values(self):
        """Non-positive or non-numeric parameter values must be rejected."""
        # Non-positive distance
        p1 = {"intent": "SEQUENTIAL", "action_sequence": [{"intent": "MOVE_FORWARD", "distance": 0.0}]}
        ok1, _, err1 = normalize_and_validate_plan(p1)
        self.assertFalse(ok1)
        self.assertIn("must be positive", err1)

        # Negative angle
        p2 = {"intent": "SEQUENTIAL", "action_sequence": [{"intent": "TURN_RIGHT", "angle": -90.0}]}
        ok2, _, err2 = normalize_and_validate_plan(p2)
        self.assertFalse(ok2)
        self.assertIn("must be positive", err2)

        # Non-numeric distance
        p3 = {"intent": "SEQUENTIAL", "action_sequence": [{"intent": "MOVE_FORWARD", "distance": "one"}]}
        ok3, _, err3 = normalize_and_validate_plan(p3)
        self.assertFalse(ok3)
        self.assertIn("must be numeric", err3)

        # Boolean distance (bool is subclass of int in Python)
        p4 = {"intent": "SEQUENTIAL", "action_sequence": [{"intent": "MOVE_FORWARD", "distance": True}]}
        ok4, _, err4 = normalize_and_validate_plan(p4)
        self.assertFalse(ok4)
        self.assertIn("must be numeric", err4)

    def test_single_target_commands_preserved(self):
        """
        Requirement 7: Do not break single-target commands.
        {"intent": "MOVE_TO_TARGET", "target": "refrigerator"} or aliases like "kitchen".
        """
        single_target = {"intent": "MOVE_TO_TARGET", "target": "kitchen"}
        ok, norm, err = normalize_and_validate_plan(single_target)
        self.assertTrue(ok, f"Validation failed: {err}")
        self.assertEqual(norm["intent"], "SEQUENTIAL")
        self.assertEqual(norm["step_count"], 1)
        self.assertEqual(norm["action_sequence"][0], {"intent": "MOVE_TO_TARGET", "target": "kitchen"})

    def test_single_action_forward_preserved(self):
        """Single MOVE_FORWARD command without action_sequence is wrapped into SEQUENTIAL."""
        single_fwd = {"intent": "MOVE_FORWARD", "distance": 1.2}
        ok, norm, err = normalize_and_validate_plan(single_fwd)
        self.assertTrue(ok, f"Validation failed: {err}")
        self.assertEqual(norm["intent"], "SEQUENTIAL")
        self.assertEqual(norm["step_count"], 1)
        self.assertEqual(norm["action_sequence"][0]["distance"], 1.2)

    def test_single_action_stop_preserved(self):
        """Single STOP command without action_sequence is wrapped into SEQUENTIAL."""
        single_stop = {"intent": "STOP"}
        ok, norm, err = normalize_and_validate_plan(single_stop)
        self.assertTrue(ok, f"Validation failed: {err}")
        self.assertEqual(norm["intent"], "SEQUENTIAL")
        self.assertEqual(norm["step_count"], 1)
        self.assertEqual(norm["action_sequence"][0], {"intent": "STOP"})

    def test_unknown_target_rejected(self):
        """Unknown location targets must be rejected."""
        plan = {
            "intent": "SEQUENTIAL",
            "action_sequence": [{"intent": "MOVE_TO_TARGET", "target": "garage"}],
        }
        ok, _, err = normalize_and_validate_plan(plan)
        self.assertFalse(ok)
        self.assertIn("unknown", err.lower())

    def test_step_count_recalculated(self):
        """step_count must reflect the actual count of actions in action_sequence."""
        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 99,  # Deliberately incorrect step_count from model
            "action_sequence": ["MOVE_FORWARD", "STOP"],
        }
        ok, norm, err = normalize_and_validate_plan(plan)
        self.assertTrue(ok, f"Validation failed: {err}")
        self.assertEqual(norm["step_count"], 2)

    def test_single_target_move_to_target_bed(self):
        """Regression Task 6: {"intent": "MOVE_TO_TARGET", "target": "bed"}."""
        payload = {"intent": "MOVE_TO_TARGET", "target": "bed"}
        ok, norm, err = normalize_and_validate_plan(payload)
        self.assertTrue(ok, f"Plan validation failed: {err}")
        self.assertEqual(norm["intent"], "SEQUENTIAL")
        self.assertEqual(norm["step_count"], 1)
        self.assertEqual(len(norm["action_sequence"]), 1)
        self.assertEqual(norm["action_sequence"][0], {"intent": "MOVE_TO_TARGET", "target": "bed"})
        self.assertEqual(norm.get("target"), "bed")

    def test_actual_move_near_target_model_output_preserved(self):
        """
        Regression Task 6: Verify the actual model output for 'বিছানার কাছে যাও':
        {
          "intent": "MOVE_NEAR_TARGET",
          "target": "bed",
          "speed_category": "NORMAL",
          "linear_speed": 0.15,
          "angular_speed": 0.0,
          "action_sequence": ["MOVE_NEAR_TARGET"],
          "step_count": 1
        }
        Must not be rejected with "Expected top-level intent 'SEQUENTIAL', got 'MOVE_NEAR_TARGET'".
        Must preserve MOVE_NEAR_TARGET intent and preserve target="bed".
        """
        actual_model_output = {
            "intent": "MOVE_NEAR_TARGET",
            "target": "bed",
            "speed_category": "NORMAL",
            "linear_speed": 0.15,
            "angular_speed": 0.0,
            "action_sequence": ["MOVE_NEAR_TARGET"],
            "step_count": 1,
        }
        ok, norm, err = normalize_and_validate_plan(actual_model_output)
        self.assertTrue(ok, f"Regression failure on actual model output: {err}")
        self.assertEqual(norm["intent"], "SEQUENTIAL")
        self.assertEqual(norm["step_count"], 1)
        self.assertEqual(len(norm["action_sequence"]), 1)
        self.assertEqual(norm["action_sequence"][0], {"intent": "MOVE_NEAR_TARGET", "target": "bed"})
        self.assertEqual(norm.get("target"), "bed")

    def test_valid_5step_sequential_plan_working_behavior(self):
        """
        Regression Task 6: Working 5-step sequential command:
        'সামনে যাও, বামে ঘুরো, সামনে যাও, ডানে ঘুরো, তারপর থামো'
        Both bare strings and structured plans must validate successfully.
        """
        # Bare-string format as produced by Qwen fine-tuned model
        bare_5step = {
            "intent": "SEQUENTIAL",
            "step_count": 5,
            "action_sequence": ["MOVE_FORWARD", "TURN_LEFT", "MOVE_FORWARD", "TURN_RIGHT", "STOP"],
        }
        ok, norm, err = normalize_and_validate_plan(bare_5step)
        self.assertTrue(ok, f"5-step bare plan failed: {err}")
        self.assertEqual(norm["intent"], "SEQUENTIAL")
        self.assertEqual(norm["step_count"], 5)
        self.assertEqual(norm["action_sequence"], [
            {"intent": "MOVE_FORWARD", "distance": 1.0},
            {"intent": "TURN_LEFT", "angle": 90.0},
            {"intent": "MOVE_FORWARD", "distance": 1.0},
            {"intent": "TURN_RIGHT", "angle": 90.0},
            {"intent": "STOP"},
        ])

        # Structured format with explicit parameters
        struct_5step = {
            "intent": "SEQUENTIAL",
            "step_count": 5,
            "action_sequence": [
                {"intent": "MOVE_FORWARD", "distance": 1.5},
                {"intent": "TURN_LEFT", "angle": 45.0},
                {"intent": "MOVE_FORWARD", "distance": 0.8},
                {"intent": "TURN_RIGHT", "angle": 60.0},
                {"intent": "STOP"},
            ],
        }
        ok2, norm2, err2 = normalize_and_validate_plan(struct_5step)
        self.assertTrue(ok2, f"5-step structured plan failed: {err2}")
        self.assertEqual(norm2["step_count"], 5)
        self.assertEqual(norm2["action_sequence"][0]["distance"], 1.5)
        self.assertEqual(norm2["action_sequence"][1]["angle"], 45.0)
        self.assertEqual(norm2["action_sequence"][2]["distance"], 0.8)
        self.assertEqual(norm2["action_sequence"][3]["angle"], 60.0)
        self.assertEqual(norm2["action_sequence"][4]["intent"], "STOP")

    def test_invalid_or_unsupported_intents_rejected(self):
        """Regression Task 6: Invalid or unsupported intents must be strictly rejected."""
        # Unsupported intent at top-level
        p1 = {"intent": "FLY_DRONE"}
        ok1, _, err1 = normalize_and_validate_plan(p1)
        self.assertFalse(ok1)
        self.assertIn("Unsupported", err1)

        # UNKNOWN_TARGET at top-level
        p2 = {"intent": "UNKNOWN_TARGET", "target": ""}
        ok2, _, err2 = normalize_and_validate_plan(p2)
        self.assertFalse(ok2)
        self.assertIn("Unsupported", err2)

        # MOVE_NEAR_TARGET without target
        p3 = {"intent": "MOVE_NEAR_TARGET"}
        ok3, _, err3 = normalize_and_validate_plan(p3)
        self.assertFalse(ok3)
        self.assertIn("requires a 'target' parameter", err3)

        # MOVE_NEAR_TARGET with unknown target
        p4 = {"intent": "MOVE_NEAR_TARGET", "target": "garage"}
        ok4, _, err4 = normalize_and_validate_plan(p4)
        self.assertFalse(ok4)
        self.assertIn("unknown", err4.lower())

        # MOVE_TO_TARGET with unknown target
        p5 = {"intent": "MOVE_TO_TARGET", "target": "mars"}
        ok5, _, err5 = normalize_and_validate_plan(p5)
        self.assertFalse(ok5)
        self.assertIn("unknown", err5.lower())

        # MOVE_TO_TARGET with empty target
        p6 = {"intent": "MOVE_TO_TARGET", "target": "   "}
        ok6, _, err6 = normalize_and_validate_plan(p6)
        self.assertFalse(ok6)
        self.assertIn("non-empty", err6.lower())

        # Sequential plan with unsupported inner action
        p7 = {"intent": "SEQUENTIAL", "action_sequence": [{"intent": "FLY_DRONE"}]}
        ok7, _, err7 = normalize_and_validate_plan(p7)
        self.assertFalse(ok7)
        self.assertIn("Unsupported action intent 'FLY_DRONE'", err7)

        # Empty action_sequence
        p8 = {"intent": "SEQUENTIAL", "action_sequence": []}
        ok8, _, err8 = normalize_and_validate_plan(p8)
        self.assertFalse(ok8)
        self.assertIn("must not be empty", err8)

    def test_action_aliases_and_allowed_actions_constants(self):
        """Verify ACTION_ALIASES and ALLOWED_ACTIONS definitions."""
        expected_aliases = {
            "MOVE_TOWARD_TARGET": "MOVE_TO_TARGET",
            "ROTATE_LEFT": "TURN_LEFT",
            "ROTATE_RIGHT": "TURN_RIGHT",
        }
        self.assertEqual(ACTION_ALIASES, expected_aliases)

        expected_allowed = {
            "MOVE_BACKWARD",
            "MOVE_FORWARD",
            "MOVE_TO_TARGET",
            "MOVE_NEAR_TARGET",
            "STOP",
            "TURN_LEFT",
            "TURN_RIGHT",
        }
        self.assertEqual(ALLOWED_ACTIONS, expected_allowed)

    def test_move_toward_target_normalization_in_sequence(self):
        """
        Verify MOVE_TOWARD_TARGET is normalized to MOVE_TO_TARGET before validation.
        Preserves the target associated with each action, original action order, and step_count.
        """
        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 2,
            "action_sequence": [
                {"intent": "MOVE_FORWARD", "distance": 1.2},
                {"intent": "MOVE_TOWARD_TARGET", "target": "table"},
            ],
        }
        ok, norm, err = normalize_and_validate_plan(plan)
        self.assertTrue(ok, f"Validation failed: {err}")
        self.assertEqual(norm["intent"], "SEQUENTIAL")
        self.assertEqual(norm["step_count"], 2)
        self.assertEqual(len(norm["action_sequence"]), 2)
        # Verify action 1
        self.assertEqual(norm["action_sequence"][0], {"intent": "MOVE_FORWARD", "distance": 1.2})
        # Verify action 2 normalized to canonical MOVE_TO_TARGET with preserved target
        self.assertEqual(norm["action_sequence"][1], {"intent": "MOVE_TO_TARGET", "target": "table"})

    def test_move_near_target_normalization_in_sequence(self):
        """
        Verify MOVE_NEAR_TARGET intent is preserved before validation.
        Preserves the target associated with each action, original action order, and step_count.
        """
        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 2,
            "action_sequence": [
                {"intent": "MOVE_NEAR_TARGET", "target": "bed"},
                {"intent": "STOP"},
            ],
        }
        ok, norm, err = normalize_and_validate_plan(plan)
        self.assertTrue(ok, f"Validation failed: {err}")
        self.assertEqual(norm["intent"], "SEQUENTIAL")
        self.assertEqual(norm["step_count"], 2)
        self.assertEqual(len(norm["action_sequence"]), 2)
        self.assertEqual(norm["action_sequence"][0], {"intent": "MOVE_NEAR_TARGET", "target": "bed"})
        self.assertEqual(norm["action_sequence"][1], {"intent": "STOP"})

    def test_mixed_sequence_with_toward_and_near_preserves_targets_and_order(self):
        """
        Verify multi-step action sequence containing MOVE_TOWARD_TARGET, MOVE_NEAR_TARGET,
        and multiple motion/special actions preserves exact order, distinct targets, and step_count.
        """
        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 6,
            "action_sequence": [
                {"intent": "MOVE_FORWARD", "distance": 1.0},
                {"intent": "MOVE_TOWARD_TARGET", "target": "refrigerator"},
                {"intent": "TURN_LEFT", "angle": 90.0},
                {"intent": "MOVE_NEAR_TARGET", "target": "sofa"},
                {"intent": "MOVE_BACKWARD", "distance": 0.5},
                {"intent": "STOP"},
            ],
        }
        ok, norm, err = normalize_and_validate_plan(plan)
        self.assertTrue(ok, f"Validation failed on mixed sequence: {err}")
        self.assertEqual(norm["intent"], "SEQUENTIAL")
        self.assertEqual(norm["step_count"], 6)
        actions = norm["action_sequence"]
        self.assertEqual(len(actions), 6)

        # Step 1: MOVE_FORWARD
        self.assertEqual(actions[0], {"intent": "MOVE_FORWARD", "distance": 1.0})
        # Step 2: MOVE_TOWARD_TARGET -> MOVE_TO_TARGET (target="refrigerator")
        self.assertEqual(actions[1], {"intent": "MOVE_TO_TARGET", "target": "refrigerator"})
        # Step 3: TURN_LEFT
        self.assertEqual(actions[2], {"intent": "TURN_LEFT", "angle": 90.0})
        # Step 4: MOVE_NEAR_TARGET is preserved (target="sofa")
        self.assertEqual(actions[3], {"intent": "MOVE_NEAR_TARGET", "target": "sofa"})
        # Step 5: MOVE_BACKWARD
        self.assertEqual(actions[4], {"intent": "MOVE_BACKWARD", "distance": 0.5})
        # Step 6: STOP
        self.assertEqual(actions[5], {"intent": "STOP"})

    def test_bare_string_action_aliases_normalization(self):
        """
        Verify string actions MOVE_TOWARD_TARGET and MOVE_NEAR_TARGET are normalized
        properly when the LLM outputs a string action_sequence.
        """
        bare_plan_toward = {
            "intent": "SEQUENTIAL",
            "step_count": 2,
            "target": "table",
            "action_sequence": ["MOVE_FORWARD", "MOVE_TOWARD_TARGET"],
        }
        ok1, norm1, err1 = normalize_and_validate_plan(bare_plan_toward)
        self.assertTrue(ok1, f"Bare MOVE_TOWARD_TARGET failed: {err1}")
        self.assertEqual(norm1["step_count"], 2)
        self.assertEqual(norm1["action_sequence"][0], {"intent": "MOVE_FORWARD", "distance": 1.0})
        self.assertEqual(norm1["action_sequence"][1], {"intent": "MOVE_TO_TARGET", "target": "table"})

        bare_plan_near = {
            "intent": "SEQUENTIAL",
            "step_count": 2,
            "target": "bed",
            "action_sequence": ["MOVE_NEAR_TARGET", "STOP"],
        }
        ok2, norm2, err2 = normalize_and_validate_plan(bare_plan_near)
        self.assertTrue(ok2, f"Bare MOVE_NEAR_TARGET failed: {err2}")
        self.assertEqual(norm2["step_count"], 2)
        self.assertEqual(norm2["action_sequence"][0], {"intent": "MOVE_NEAR_TARGET", "target": "bed"})
        self.assertEqual(norm2["action_sequence"][1], {"intent": "STOP"})


    def test_single_action_move_toward_target(self):
        """Verify single-action MOVE_TOWARD_TARGET wraps to SEQUENTIAL with canonical MOVE_TO_TARGET."""
        payload = {"intent": "MOVE_TOWARD_TARGET", "target": "table"}
        ok, norm, err = normalize_and_validate_plan(payload)
        self.assertTrue(ok, f"Single MOVE_TOWARD_TARGET failed: {err}")
        self.assertEqual(norm["intent"], "SEQUENTIAL")
        self.assertEqual(norm["step_count"], 1)
        self.assertEqual(norm["action_sequence"][0], {"intent": "MOVE_TO_TARGET", "target": "table"})
        self.assertEqual(norm.get("target"), "table")

    def test_move_toward_target_error_cases(self):
        """Verify MOVE_TOWARD_TARGET rejects missing or unknown targets."""
        # Missing target
        p1 = {"intent": "MOVE_TOWARD_TARGET"}
        ok1, _, err1 = normalize_and_validate_plan(p1)
        self.assertFalse(ok1)
        self.assertIn("requires a 'target' parameter", err1)

        # Empty target
        p2 = {"intent": "MOVE_TOWARD_TARGET", "target": "  "}
        ok2, _, err2 = normalize_and_validate_plan(p2)
        self.assertFalse(ok2)
        self.assertIn("non-empty", err2.lower())

        # Unknown target
        p3 = {"intent": "MOVE_TOWARD_TARGET", "target": "spaceship"}
        ok3, _, err3 = normalize_and_validate_plan(p3)
        self.assertFalse(ok3)
        self.assertIn("unknown", err3.lower())

        # In sequence with unknown target
        p4 = {
            "intent": "SEQUENTIAL",
            "action_sequence": [{"intent": "MOVE_TOWARD_TARGET", "target": "garage"}],
        }
        ok4, _, err4 = normalize_and_validate_plan(p4)
        self.assertFalse(ok4)
        self.assertIn("unknown", err4.lower())

    def test_all_canonical_allowed_actions_remain_valid(self):
        """Verify all 6 ALLOWED_ACTIONS remain valid in sequences."""
        plan = {
            "intent": "SEQUENTIAL",
            "step_count": 6,
            "action_sequence": [
                {"intent": "MOVE_FORWARD", "distance": 1.5},
                {"intent": "MOVE_BACKWARD", "distance": 0.8},
                {"intent": "TURN_LEFT", "angle": 45.0},
                {"intent": "TURN_RIGHT", "angle": 30.0},
                {"intent": "MOVE_TO_TARGET", "target": "table"},
                {"intent": "STOP"},
            ],
        }
        ok, norm, err = normalize_and_validate_plan(plan)
        self.assertTrue(ok, f"Allowed actions validation failed: {err}")
        self.assertEqual(norm["step_count"], 6)
        intents = [a["intent"] for a in norm["action_sequence"]]
        self.assertEqual(
            intents,
            ["MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", "MOVE_TO_TARGET", "STOP"],
        )



# ===========================================================================
# 4. End-to-End Simulation of the Problem Input
# ===========================================================================
class TestProblemCommandSimulation(unittest.TestCase):
    """
    Simulates the exact user problem command:
    Input: "সামনে যাও, বামে ঘুরো, তারপর থামো"
    """

    def test_exact_input_sequential_simulation(self):
        """
        Verify end-to-end processing for "সামনে যাও, বামে ঘুরো, তারপর থামো".
        Model outputs SEQUENTIAL plan followed by repeated step decompositions:
        {"intent": "SEQUENTIAL", "step_count": 3, "action_sequence": ["MOVE_FORWARD", "TURN_LEFT", "STOP"]}
        {"intent": "MOVE_FORWARD"}
        {"intent": "TURN_LEFT"}
        {"intent": "STOP"}
        Must yield exactly ONE valid normalized JSON object.
        """
        raw_output = (
            '{"intent": "SEQUENTIAL", "step_count": 3, "action_sequence": ["MOVE_FORWARD", "TURN_LEFT", "STOP"]}\n'
            '{"intent": "MOVE_FORWARD"}\n'
            '{"intent": "TURN_LEFT"}\n'
            '{"intent": "STOP"}'
        )

        parsed_json, json_ok = parse_llm_output(raw_output)
        self.assertTrue(json_ok, "Failed to decode JSON from model output.")

        val_ok, norm_plan, val_err = normalize_and_validate_plan(parsed_json)
        self.assertTrue(val_ok, f"Plan validation failed: {val_err}")

        # Ensure exact single JSON structure
        self.assertEqual(norm_plan["intent"], "SEQUENTIAL")
        self.assertEqual(norm_plan["step_count"], 3)
        self.assertEqual(norm_plan["action_sequence"], [
            {"intent": "MOVE_FORWARD", "distance": 1.0},
            {"intent": "TURN_LEFT", "angle": 90.0},
            {"intent": "STOP"},
        ])

        # Verify serializability into exactly one valid JSON object
        final_str = json.dumps(norm_plan, ensure_ascii=False)
        reloaded = json.loads(final_str)
        self.assertIsInstance(reloaded, dict)
        self.assertEqual(reloaded["step_count"], 3)


# ===========================================================================
# 5. Tests for Prompt Construction & Chat Template Handling
# ===========================================================================
class TestPromptConstruction(unittest.TestCase):
    """
    Regression and unit tests verifying that prompt construction works when
    tokenizer.chat_template is None, when tokenizer is absent, and that it
    strictly adheres to the ChatML fine-tuning format and system prompt.
    """

    def test_format_prompt_when_chat_template_is_none_regression(self):
        """
        Regression: When tokenizer.chat_template is None, format_prompt must NOT
        raise ValueError and must output the exact ChatML prompt format matching fine-tuning.
        """
        class TokenizerWithNoneChatTemplate:
            def __init__(self):
                self.chat_template = None

            def apply_chat_template(self, messages, chat_template=None, tokenize=False, add_generation_prompt=False):
                template = chat_template or self.chat_template
                if template is None:
                    raise ValueError(
                        "Cannot use chat template functions because tokenizer.chat_template "
                        "is not set and no template argument was passed!"
                    )
                res = ""
                for msg in messages:
                    res += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                if add_generation_prompt:
                    res += "<|im_start|>assistant\n"
                return res

        tok = TokenizerWithNoneChatTemplate()
        self.assertIsNone(tok.chat_template)

        cmd = "সামনে যাও, বামে ঘুরো, তারপর থামো"
        prompt = format_prompt(cmd, tokenizer=tok)

        expected = (
            "<|im_start|>system\n"
            "You are a robot command mapping assistant. Convert the given Bangla voice command into a structured JSON action plan.<|im_end|>\n"
            "<|im_start|>user\n"
            "সামনে যাও, বামে ঘুরো, তারপর থামো<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.assertEqual(prompt, expected)

    def test_format_prompt_when_tokenizer_is_none(self):
        """When tokenizer is None, format_prompt constructs the exact ChatML prompt directly."""
        cmd = "টেবিলে যাও"
        prompt = format_prompt(cmd, tokenizer=None)
        expected = (
            "<|im_start|>system\n"
            "You are a robot command mapping assistant. Convert the given Bangla voice command into a structured JSON action plan.<|im_end|>\n"
            "<|im_start|>user\n"
            "টেবিলে যাও<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.assertEqual(prompt, expected)

    def test_format_prompt_with_mock_tokenizer_without_chat_template_attribute(self):
        """When a tokenizer object has chat_template = None and no apply_chat_template, it formats correctly."""
        tok = MockTokenizer()
        tok.chat_template = None
        cmd = "পিছনে যাও"
        prompt = format_prompt(cmd, tokenizer=tok)
        expected = (
            "<|im_start|>system\n"
            "You are a robot command mapping assistant. Convert the given Bangla voice command into a structured JSON action plan.<|im_end|>\n"
            "<|im_start|>user\n"
            "পিছনে যাও<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.assertEqual(prompt, expected)

    def test_format_prompt_preserves_default_system_prompt(self):
        """Preserves the exact Experiment 1 fine-tuning system prompt."""
        self.assertEqual(
            DEFAULT_SYSTEM_PROMPT,
            "You are a robot command mapping assistant. Convert the given Bangla voice command into a structured JSON action plan."
        )

    def test_format_prompt_custom_system_prompt(self):
        """Custom system prompts are correctly injected into the ChatML system turn."""
        custom_sys = "Custom robot controller prompt."
        cmd = "থামো"
        prompt = format_prompt(cmd, system_prompt=custom_sys, tokenizer=None)
        self.assertTrue(prompt.startswith(f"<|im_start|>system\n{custom_sys}<|im_end|>\n"))
        self.assertIn(f"<|im_start|>user\n{cmd}<|im_end|>\n", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_configure_chat_template_sets_template_when_none(self):
        """configure_chat_template explicitly assigns QWEN_CHAT_TEMPLATE when tokenizer.chat_template is None."""
        class DummyTokenizer:
            def __init__(self):
                self.chat_template = None

        tok = DummyTokenizer()
        self.assertIsNone(tok.chat_template)
        configure_chat_template(tok)
        self.assertIsNotNone(tok.chat_template)
        self.assertEqual(tok.chat_template, QWEN_CHAT_TEMPLATE)

    def test_configure_chat_template_preserves_existing_template(self):
        """configure_chat_template preserves any already configured chat_template."""
        class DummyTokenizer:
            def __init__(self):
                self.chat_template = "EXISTING_TEMPLATE_CONTENT"

        tok = DummyTokenizer()
        configure_chat_template(tok)
        self.assertEqual(tok.chat_template, "EXISTING_TEMPLATE_CONTENT")


class TestMultiTargetSequentialFix(unittest.TestCase):
    """
    Test suite for sequential multi-target navigation bug fix.
    Validates:
    - Test A: Single-target command continues working.
    - Test B: Two-target sequential command targets bedroom then table distinctly.
    - Test C: Different sequential targets (refrigerator then sofa).
    - Test D: Action intent preservation (MOVE_NEAR_TARGET preserved, not converted to MOVE_TO_TARGET).
    - Explicit rejection of unsupported targets (never silently replaced with table).
    - Prevention of duplicate target assignment when distinct targets cannot be resolved.
    """

    def test_target_extraction_from_command(self):
        """Verify extract_targets_from_command accurately extracts sequential targets."""
        # Single target
        self.assertEqual(extract_targets_from_command("রোবট, টেবিলের কাছে যাও।"), ["table"])
        # Two targets: bedroom, table
        self.assertEqual(
            extract_targets_from_command("রোবট, বেডরুমে যাও, তারপর টেবিলের কাছে যাও।"),
            ["bedroom", "table"]
        )
        # Two targets: refrigerator, sofa
        self.assertEqual(
            extract_targets_from_command("রোবট, রেফ্রিজারেটরের কাছে যাও, তারপর সোফার কাছে যাও।"),
            ["refrigerator", "sofa"]
        )
        # Bedroom variations
        self.assertEqual(extract_targets_from_command("শোবার ঘরে যাও"), ["bedroom"])
        self.assertEqual(extract_targets_from_command("খাটের কাছে যাও"), ["bed"])
        # Table variations
        self.assertEqual(extract_targets_from_command("ডাইনিং টেবিলে যাও"), ["dining_table"])
        # Unsupported location
        self.assertEqual(
            extract_targets_from_command("রোবট, বাথরুমে যাও, তারপর টেবিলের কাছে যাও।"),
            ["bathroom", "table"]
        )

    def test_test_a_existing_single_target_command(self):
        """
        Test A — Existing single-target command: 'রোবট, টেবিলের কাছে যাও।'
        Expected: It continues working exactly as before.
        """
        cmd = "রোবট, টেবিলের কাছে যাও।"
        # Model output for single command
        model_output = {
            "intent": "MOVE_NEAR_TARGET",
            "target": "table",
            "speed_category": "NORMAL",
            "linear_speed": 0.15,
            "angular_speed": 0.0,
            "action_sequence": ["MOVE_NEAR_TARGET"],
            "step_count": 1,
        }
        ok, norm_plan, err = normalize_and_validate_plan(model_output, command=cmd)
        self.assertTrue(ok, f"Test A failed: {err}")
        self.assertEqual(norm_plan["intent"], "SEQUENTIAL")
        self.assertEqual(norm_plan["step_count"], 1)
        self.assertEqual(len(norm_plan["action_sequence"]), 1)
        self.assertEqual(norm_plan["action_sequence"][0]["target"], "table")
        self.assertEqual(norm_plan.get("target"), "table")

    def test_test_b_two_target_sequential_command(self):
        """
        Test B — Two-target sequential command: 'রোবট, বেডরুমে যাও, তারপর টেবিলের কাছে যাও।'
        Expected:
        - Step 1 targets the correct bedroom-related location.
        - Step 2 targets the table.
        - The two steps must not silently use the same target.
        """
        cmd = "রোবট, বেডরুমে যাও, তারপর টেবিলের কাছে যাও।"
        # Exact raw output from fine-tuned LLM reported by user
        raw_llm_output = {
            "intent": "SEQUENTIAL",
            "target": "table",
            "speed_category": "NORMAL",
            "linear_speed": 0.15,
            "angular_speed": 0.0,
            "action_sequence": [
                "MOVE_TO_TARGET",
                "MOVE_NEAR_TARGET"
            ],
            "step_count": 2
        }
        ok, norm_plan, err = normalize_and_validate_plan(raw_llm_output, command=cmd)
        self.assertTrue(ok, f"Test B failed: {err}")
        self.assertEqual(norm_plan["intent"], "SEQUENTIAL")
        self.assertEqual(norm_plan["step_count"], 2)
        actions = norm_plan["action_sequence"]
        self.assertEqual(len(actions), 2)

        # Step 1 targets bedroom (legacy alias of bed in locations.yaml)
        self.assertEqual(actions[0]["intent"], "MOVE_TO_TARGET")
        self.assertEqual(actions[0]["target"], "bedroom")

        # Step 2 targets table
        self.assertEqual(actions[1]["intent"], "MOVE_NEAR_TARGET")
        self.assertEqual(actions[1]["target"], "table")

        # Must not silently use the same target
        self.assertNotEqual(actions[0]["target"], actions[1]["target"])

    def test_test_c_different_sequential_targets(self):
        """
        Test C — Different sequential targets: 'রোবট, রেফ্রিজারেটরের কাছে যাও, তারপর সোফার কাছে যাও।'
        Expected:
        - Step 1 targets refrigerator.
        - Step 2 targets sofa.
        """
        cmd = "রোবট, রেফ্রিজারেটরের কাছে যাও, তারপর সোফার কাছে যাও।"
        raw_llm_output = {
            "intent": "SEQUENTIAL",
            "target": "sofa",
            "speed_category": "NORMAL",
            "linear_speed": 0.15,
            "angular_speed": 0.0,
            "action_sequence": [
                "MOVE_NEAR_TARGET",
                "MOVE_NEAR_TARGET"
            ],
            "step_count": 2
        }
        ok, norm_plan, err = normalize_and_validate_plan(raw_llm_output, command=cmd)
        self.assertTrue(ok, f"Test C failed: {err}")
        self.assertEqual(norm_plan["intent"], "SEQUENTIAL")
        self.assertEqual(norm_plan["step_count"], 2)
        actions = norm_plan["action_sequence"]
        self.assertEqual(len(actions), 2)

        # Step 1 targets refrigerator
        self.assertEqual(actions[0]["intent"], "MOVE_NEAR_TARGET")
        self.assertEqual(actions[0]["target"], "refrigerator")

        # Step 2 targets sofa
        self.assertEqual(actions[1]["intent"], "MOVE_NEAR_TARGET")
        self.assertEqual(actions[1]["target"], "sofa")

        # Distinct targets
        self.assertNotEqual(actions[0]["target"], actions[1]["target"])

    def test_test_d_action_intent_preservation(self):
        """
        Test D — Action intent preservation:
        Check that MOVE_NEAR_TARGET remains correctly represented and is not unintentionally converted to MOVE_TO_TARGET.
        """
        cmd = "রোবট, বেডরুমে যাও, তারপর টেবিলের কাছে যাও।"
        raw_llm_output = {
            "intent": "SEQUENTIAL",
            "target": "table",
            "action_sequence": [
                "MOVE_TO_TARGET",
                "MOVE_NEAR_TARGET"
            ],
            "step_count": 2
        }
        ok, norm_plan, err = normalize_and_validate_plan(raw_llm_output, command=cmd)
        self.assertTrue(ok, f"Test D failed: {err}")
        actions = norm_plan["action_sequence"]

        # Step 1 is MOVE_TO_TARGET
        self.assertEqual(actions[0]["intent"], "MOVE_TO_TARGET")
        # Step 2 must be preserved as MOVE_NEAR_TARGET (not converted to MOVE_TO_TARGET)
        self.assertEqual(actions[1]["intent"], "MOVE_NEAR_TARGET")
        self.assertNotEqual(actions[1]["intent"], "MOVE_TO_TARGET")

    def test_unsupported_target_handled_explicitly(self):
        """
        If bedroom or any target is not recognized, handle the unsupported target explicitly
        rather than silently replacing it with table.
        """
        cmd = "রোবট, বাথরুমে যাও, তারপর টেবিলের কাছে যাও।"
        raw_llm_output = {
            "intent": "SEQUENTIAL",
            "target": "table",
            "action_sequence": [
                "MOVE_TO_TARGET",
                "MOVE_NEAR_TARGET"
            ],
            "step_count": 2
        }
        ok, norm_plan, err = normalize_and_validate_plan(raw_llm_output, command=cmd)
        # Must fail explicitly because bathroom is unknown; never silently navigate to table twice
        self.assertFalse(ok)
        self.assertIn("bathroom", err.lower())
        self.assertIn("unknown", err.lower())

    def test_multi_target_duplicate_fallback_prevented(self):
        """
        If sequential action sequence contains multiple target actions but distinct targets cannot
        be resolved, reject rather than silently assigning the same top_target to both.
        """
        raw_llm_output = {
            "intent": "SEQUENTIAL",
            "target": "table",
            "action_sequence": [
                "MOVE_TO_TARGET",
                "MOVE_NEAR_TARGET"
            ],
            "step_count": 2
        }
        # No command provided, cannot resolve distinct targets
        ok, norm_plan, err = normalize_and_validate_plan(raw_llm_output, command="")
        self.assertFalse(ok)
        self.assertIn("distinct targets", err.lower())


# ===========================================================================
# Bangla Distance, Rotation & Target Resolution Tests
# ===========================================================================
class TestBanglaDistanceAndRotationFixes(unittest.TestCase):
    """Verify Issue 1, 2, and 5 fixes in interactive_inference_fixed."""

    def test_bangla_digit_and_number_parsing(self):
        """Test parse_bangla_or_english_number with various formats."""
        self.assertEqual(parse_bangla_or_english_number("১.৫"), 1.5)
        self.assertEqual(parse_bangla_or_english_number("২"), 2.0)
        self.assertEqual(parse_bangla_or_english_number("০.৫"), 0.5)
        self.assertEqual(parse_bangla_or_english_number("1.5"), 1.5)
        self.assertEqual(parse_bangla_or_english_number("2"), 2.0)
        self.assertEqual(parse_bangla_or_english_number("0.5"), 0.5)
        self.assertEqual(parse_bangla_or_english_number("৯০"), 90.0)
        self.assertEqual(parse_bangla_or_english_number("১৮০"), 180.0)
        self.assertIsNone(parse_bangla_or_english_number("abc"))
        self.assertIsNone(parse_bangla_or_english_number(None))

    def test_extract_distances_and_angles(self):
        """Test extraction of explicit distances and angles from Bangla commands."""
        cmd16 = "রোবট, সামনে ১.৫ মিটার যাও, বামে ৯০ ডিগ্রি ঘোরো, সামনে ২ মিটার যাও, তারপর ডানে ৯০ ডিগ্রি ঘোরো।"
        dists16 = extract_distances_from_command(cmd16)
        angles16 = extract_angles_from_command(cmd16)
        self.assertEqual(dists16, [1.5, 2.0])
        self.assertEqual(angles16, [90.0, 90.0])

        cmd3 = "রোবট, সামনে ১ মিটার যাও, তারপর পিছনে ০.৫ মিটার যাও।"
        dists3 = extract_distances_from_command(cmd3)
        self.assertEqual(dists3, [1.0, 0.5])

        cmd5 = "রোবট, ডানে ১৮০ ডিগ্রি ঘোরো, তারপর সামনে ১ মিটার যাও।"
        dists5 = extract_distances_from_command(cmd5)
        angles5 = extract_angles_from_command(cmd5)
        self.assertEqual(dists5, [1.0])
        self.assertEqual(angles5, [180.0])

    def test_rotate_right_and_rotate_left_mapping(self):
        """Verify ROTATE_LEFT -> TURN_LEFT and ROTATE_RIGHT -> TURN_RIGHT."""
        # Single action dict
        plan_left = {"intent": "ROTATE_LEFT", "angle": 90.0}
        ok, res, _ = normalize_and_validate_plan(plan_left)
        self.assertTrue(ok)
        self.assertEqual(res["action_sequence"][0]["intent"], "TURN_LEFT")
        self.assertEqual(res["action_sequence"][0]["angle"], 90.0)

        plan_right = {"intent": "ROTATE_RIGHT", "angle": 45.0}
        ok, res, _ = normalize_and_validate_plan(plan_right)
        self.assertTrue(ok)
        self.assertEqual(res["action_sequence"][0]["intent"], "TURN_RIGHT")
        self.assertEqual(res["action_sequence"][0]["angle"], 45.0)

        # Sequence of bare strings
        seq_plan = {
            "intent": "SEQUENTIAL",
            "step_count": 2,
            "action_sequence": ["ROTATE_LEFT", "ROTATE_RIGHT"]
        }
        ok, res, _ = normalize_and_validate_plan(seq_plan)
        self.assertTrue(ok)
        self.assertEqual(res["action_sequence"][0]["intent"], "TURN_LEFT")
        self.assertEqual(res["action_sequence"][1]["intent"], "TURN_RIGHT")

    def test_preservation_of_explicit_distances_in_plan(self):
        """Explicit distances like 1.5m and 2m must be preserved from voice commands."""
        cmd = "রোবট, সামনে ১.৫ মিটার যাও, বামে ৯০ ডিগ্রি ঘোরো, সামনে ২ মিটার যাও, তারপর ডানে ৯০ ডিগ্রি ঘোরো।"
        raw_plan = {
            "intent": "SEQUENTIAL",
            "step_count": 4,
            "action_sequence": [
                "MOVE_FORWARD",
                "TURN_LEFT",
                "MOVE_FORWARD",
                "TURN_RIGHT"
            ]
        }
        ok, res, err = normalize_and_validate_plan(raw_plan, command=cmd)
        self.assertTrue(ok, f"Validation failed: {err}")
        seq = res["action_sequence"]
        self.assertEqual(seq[0]["intent"], "MOVE_FORWARD")
        self.assertEqual(seq[0]["distance"], 1.5)
        self.assertEqual(seq[1]["intent"], "TURN_LEFT")
        self.assertEqual(seq[1]["angle"], 90.0)
        self.assertEqual(seq[2]["intent"], "MOVE_FORWARD")
        self.assertEqual(seq[2]["distance"], 2.0)
        self.assertEqual(seq[3]["intent"], "TURN_RIGHT")
        self.assertEqual(seq[3]["angle"], 90.0)

    def test_target_aliases_bedroom_and_kitchen(self):
        """Verify bedroom -> bed and kitchen -> refrigerator resolution in plan and database."""
        cmd = "রোবট, রান্নাঘরে যাও, তারপর বেডরুমে যাও, শেষে টেবিলের কাছে যাও।"
        raw_plan = {
            "intent": "SEQUENTIAL",
            "step_count": 3,
            "action_sequence": [
                "MOVE_TO_TARGET",
                "MOVE_TO_TARGET",
                "MOVE_TO_TARGET"
            ]
        }
        ok, res, err = normalize_and_validate_plan(raw_plan, command=cmd)
        self.assertTrue(ok, f"Validation failed: {err}")
        seq = res["action_sequence"]
        self.assertEqual(seq[0]["target"], "kitchen")
        self.assertEqual(seq[1]["target"], "bedroom")
        self.assertEqual(seq[2]["target"], "table")

        # Verify against LocationDatabase resolution
        from location_navigation.location_database import LocationDatabase, TARGET_ALIASES
        db = LocationDatabase(os.path.join(BASE_DIR, "src", "location_navigation", "config", "locations.yaml"))
        self.assertEqual(TARGET_ALIASES["kitchen"], "refrigerator")
        self.assertEqual(TARGET_ALIASES["bedroom"], "bed")
        self.assertEqual(db.get(seq[0]["target"])["x"], db.get("refrigerator")["x"])
        self.assertEqual(db.get(seq[1]["target"])["x"], db.get("bed")["x"])
        self.assertEqual(db.get(seq[2]["target"])["x"], db.get("table")["x"])


if __name__ == "__main__":
    unittest.main()

