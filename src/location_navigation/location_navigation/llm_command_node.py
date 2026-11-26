#!/usr/bin/env python3
import os
import sys
import json
import rclpy
from rclpy.node import Node
import subprocess

# Processes the user's command using Qwen.
# Validates the generated JSON plan.
# Sends the plan to llm_to_nav.
# Executes the robot's navigation.

def process_command(command):
    print(f"\nProcessing command: {command}")

    # Check for Agent Target Query (e.g. "Where is the table?", "রান্নাঘর কোথায়?")
    cmd_lower = command.strip().lower()
    is_target_query = any(w in cmd_lower for w in ["where is", "where's", "find", "locate", "কোথায়", "কোথায়", "কোন দিকে"])
    if is_target_query:
        try:
            from location_navigation.agent_target_query import query_and_select_target
            query_res = query_and_select_target(command)
            if query_res is not None:
                return
        except Exception as e:
            print(f"[llm_command_node] Target query resolution error: {e}")
    
    qwen_python = os.path.expanduser("~/case_study/llm/QWEN_LINUX_TRANSFER/.venv/bin/python")
    inference_script = os.path.expanduser("~/case_study/src/location_navigation/location_navigation/qwen_inference.py")
    
    if not os.path.exists(qwen_python):
        print(f"Error: Qwen Python environment not found at {qwen_python}")
        return
        
    if not os.path.exists(inference_script):
        print(f"Error: Inference script not found at {inference_script}")
        return
        
    print("Delegating to Qwen subprocess...", flush=True)
    try:
        result = subprocess.run(
            [qwen_python, inference_script, command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # stderr contains model loading logs, stdout contains only JSON
        if result.stderr:
            print(result.stderr, end='')
            
        json_output = result.stdout.strip()
        parsed_json = json.loads(json_output)
    except subprocess.CalledProcessError as e:
        print(f"Error running LLM subprocess: {e}")
        if e.stderr:
            print(e.stderr)
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from LLM: {e}\nRaw output: {result.stdout}")
        parsed_json = {"intent": "UNKNOWN_TARGET", "target": ""}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

    intent = parsed_json.get("intent", "")
    target = parsed_json.get("target", "")
    allowed_targets = [
        "kitchen", "bedroom", "tv", "dining_table",
        "refrigerator", "bed", "sofa", "table"
    ]

    # Validate intent
    if intent == "SEQUENTIAL":
        if "action_sequence" not in parsed_json or not parsed_json.get("action_sequence"):
            print("Invalid SEQUENTIAL plan: missing or empty action_sequence.")
            parsed_json = {"intent": "UNKNOWN_TARGET", "target": ""}
    elif intent in ["MOVE_TO_TARGET", "MOVE_NEAR_TARGET"]:
        if intent == "MOVE_NEAR_TARGET":
            intent = "MOVE_TO_TARGET"
            parsed_json["intent"] = "MOVE_TO_TARGET"
        if target not in allowed_targets:
            print(f"Unknown target ({target}). Allowed targets: {allowed_targets}")
            parsed_json = {"intent": "UNKNOWN_TARGET", "target": ""}
    elif intent in ["MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]:
        # Safe single action passed to llm_to_nav
        pass
    else:
        print(f"Invalid intent ({intent}).")
        parsed_json = {"intent": "UNKNOWN_TARGET", "target": ""}
    
    json_str = json.dumps(parsed_json)
    print(f"LLM JSON Output: {json_str}")

    if parsed_json.get("intent") == "UNKNOWN_TARGET":
        print("Safe error: UNKNOWN_TARGET. Aborting navigation.")
        return

    nav_cmd = ["ros2", "run", "location_navigation", "llm_to_nav", json_str]
    for arg in sys.argv:
        if "use_sim_time:=" in arg:
            val = arg.split("use_sim_time:=")[-1].strip().lower()
            if val in ("1", "true", "yes"):
                nav_cmd.extend(["--ros-args", "-p", "use_sim_time:=true"])
            break
    else:
        if os.environ.get("USE_SIM_TIME", "").strip().lower() in ("1", "true", "yes"):
            nav_cmd.extend(["--ros-args", "-p", "use_sim_time:=true"])

    try:
        print("Delegating to llm_to_nav...")
        subprocess.run(nav_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error calling llm_to_nav: {e}")
    except FileNotFoundError:
        print("ros2 command not found. Are you in a sourced ROS 2 environment?")

def main(args=sys.argv):
    rclpy.init(args=args)
    
    from rclpy.utilities import remove_ros_args
    clean_args = remove_ros_args(args)[1:]
    
    if clean_args:
        command = clean_args[0]
        process_command(command)
    else:
        print("\nEntering interactive mode. Type 'exit' to quit.")
        while rclpy.ok():
            try:
                cmd = input("\nEnter Bangla or English command: ").strip()
                if cmd.lower() == 'exit':
                    break
                if not cmd:
                    continue
                process_command(cmd)
            except KeyboardInterrupt:
                break
            except EOFError:
                break

    rclpy.shutdown()

if __name__ == '__main__':
    main()
