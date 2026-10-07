# Location Navigation for car_bot

This package provides a natural-language to Nav2 location-based navigation bridge.

## Setup & Build

1. Source your ROS 2 environment:
   ```bash
   source /opt/ros/jazzy/setup.bash
   ```
2. Build this specific package:
   ```bash
   cd ~/case_study
   colcon build --packages-select location_navigation --symlink-install
   ```
3. Source the workspace:
   ```bash
   source ~/case_study/install/setup.bash
   ```

## Start Existing Gazebo + Nav2

Ensure the existing robot simulation and navigation stack is running:
*(Use your existing launch commands, e.g., `ros2 launch car_bot final_launch.py`)*

Verify that `/navigate_to_pose` is available:
```bash
ros2 action list | grep navigate_to_pose
```

## Manual JSON Testing

You can test the pipeline using manual JSON commands without starting the LLM.

**Test Kitchen:**
```bash
ros2 run location_navigation llm_to_nav '{"intent":"MOVE_TO_TARGET","target":"kitchen"}'
```

**Test Bedroom:**
```bash
ros2 run location_navigation llm_to_nav '{"intent":"MOVE_TO_TARGET","target":"bedroom"}'
```

**Test TV:**
```bash
ros2 run location_navigation llm_to_nav '{"intent":"MOVE_TO_TARGET","target":"tv"}'
```

**Test Dining Table:**
```bash
ros2 run location_navigation llm_to_nav '{"intent":"MOVE_TO_TARGET","target":"dining_table"}'
```

## Qwen LLM Integration

The Qwen model is integrated via the inference script. The script takes Bangla input, converts it to JSON, and automatically invokes this ROS 2 node.

To run the interactive model:
```bash
cd ~/case_study/llm/QWEN_LINUX_TRANSFER/inference
# Activate your python env with PyTorch if needed
python3 interactive_inference_fixed.py
```
*(When it outputs a valid intent to move to `kitchen`, it will automatically trigger the ROS 2 node to send the Nav2 goal.)*
