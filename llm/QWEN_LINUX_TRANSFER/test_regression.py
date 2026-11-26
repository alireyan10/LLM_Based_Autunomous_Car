#!/usr/bin/env python3
"""
Regression test runner for QWEN_LINUX_TRANSFER inference.
Executes the comprehensive test suite to verify:
1. Stopping criteria behavior (JsonStoppingCriteria).
2. LLM output parsing (parse_llm_output) on single JSON, repeated fragments, and malformed inputs.
3. Plan validation and normalization (normalize_and_validate_plan).
4. Bare-string action_sequence backward compatibility.
5. Single-target and sequential navigation preservation.
"""

import os
import sys
import unittest

def main():
    test_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "location_navigation", "test")
    )
    print(f"Discovering and running tests in: {test_dir}\n")
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=test_dir, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()
