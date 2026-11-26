# ROS 2 Jazzy + Gazebo Harmonic Mobile Robot Project

## 1. Project Overview
This project simulates a differential-drive mobile robot operating in a residential environment. 
Built on **ROS 2 Jazzy** and **Gazebo Harmonic**, the simulation provides a complete pipeline for mapping, localization, and autonomous navigation. The robot utilizes the Nav2 stack and Adaptive Monte Carlo Localization (AMCL) to navigate the `small_house` environment, dynamically detecting and avoiding obstacles using a simulated GPU LiDAR.

## 2. System Requirements
- **OS**: Ubuntu 24.04
- **ROS Version**: ROS 2 Jazzy
- **Simulation**: Gazebo Harmonic
- **Required ROS 2 Packages**: 
  - `rclpy`
  - `geometry_msgs`
  - `joint_state_publisher`, `joint_state_publisher_gui`
  - `xacro`
  - `controller_manager`, `ros2_control`, `ros2_controllers`
  - `teleop_twist_keyboard`
  - `ros_gz_sim`, `ros_gz_bridge`, `gz_ros2_control`
  - `slam_toolbox`
  - `nav2_bringup`
  - `twist_mux`

## 3. Project Directory Structure
```text
~/case_study/
├── src/
│   ├── ros2-diff-drive-robot/ (Main repository)
│   │   ├── car_bot/          (Python package files)
│   │   ├── config/           (Nav2 params, RViz configs, etc.)
│   │   ├── launch/           (ROS 2 launch files)
│   │   ├── map/              (Default/save maps)
│   │   ├── urdf/             (Robot Xacro descriptions)
│   │   └── package.xml       (Package dependencies)
│   └── location_navigation/  (LLM Navigation Interface)
├── small_house/
│   ├── small_house.world     (Main simulation world)
│   ├── models/               (Original environment models)
│   └── models_static_test/   (FINAL static environment models for performance)
├── room_map.yaml             (Final saved map configuration)
├── room_map.pgm              (Final saved map image)
├── install/                  (Colcon build output)
├── build/                    (Colcon build temporary files)
└── log/
```

## 4. Build Instructions
To build the workspace from scratch, use the following commands:
```bash
cd ~/case_study
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 5. Environment / Gazebo Resource Path
Before launching the simulation, you must set the Gazebo resource path to ensure the correct models are loaded:
```bash
export GZ_SIM_RESOURCE_PATH=$HOME/case_study/small_house/models_static_test:$HOME/case_study/small_house:$HOME/case_study/small_house/maps
```
**Model Paths Explained**:
- **Original models** (`$HOME/case_study/small_house/models`): The default unoptimized residential props with dynamic physics.
- **Final static-test models** (`$HOME/case_study/small_house/models_static_test`): Contains 29 models with `<static>true</static>` explicitly added. **The final navigation configuration must use this path** to maintain an RTF of ~1.0.

## 6. Launching the Final Simulation
Run the complete simulation stack using:
```bash
cd ~/case_study
source /opt/ros/jazzy/setup.bash
source install/setup.bash

export GZ_SIM_RESOURCE_PATH=$HOME/case_study/small_house/models_static_test:$HOME/case_study/small_house:$HOME/case_study/small_house/maps

ros2 launch car_bot final_launch.py \
map:=/home/reyan/case_study/room_map.yaml \
world:=/home/reyan/case_study/small_house/small_house.world \
use_sim_time:=true
```
- `map`: Specifies the absolute path to the preserved room map, correcting previous map-server errors.
- `world`: Loads the designated Gazebo simulation world.
- `use_sim_time`: Forces all ROS 2 nodes to sync their clocks with the Gazebo simulation.

## 7. RViz Navigation
Upon launching, RViz will open automatically.
- **Fixed Frame**: Ensure this is set to `map`.
- **2D Pose Estimate**: Use this tool to initialize the AMCL particle filter by clicking and dragging on the map where the robot is currently spawned.
- **Nav2 Goal**: Use this tool to command the robot to drive autonomously to a target pose.

## 8. Teleoperation
To drive the robot manually via keyboard, open a new sourced terminal and run:
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p stamped:=true
```
*Note: `stamped:=true` is strictly required as the robot's `twist_mux` / controller setup expects `TwistStamped` messages rather than standard `Twist` messages.*

## 9. Map
The final working environment map consists of:
- `~/case_study/room_map.yaml`
- `~/case_study/room_map.pgm`
The map is loaded by the `nav2_map_server`. The map is **NOT** regenerated during normal navigation operations. Passing `map:=/home/reyan/case_study/room_map.yaml` to `final_launch.py` guarantees this final map is published.

## 10. Localization
- **map_server**: Publishes the static map image to the `/map` topic.
- **AMCL**: Consumes the `/map` and LiDAR `/scan` to probabilistically estimate the robot's pose.
- **TF Tree**: AMCL publishes the `map` -> `odom` transform. The `ros2_control` differential drive controller handles `odom` -> `base_link`.

## 11. Navigation
The project uses the ROS 2 **Nav2** stack. Goals provided by RViz (or the Qwen LLM interface) are processed by the Nav2 Behavior Server. Path planning is executed on the global costmap using the `GridBased` planner, while the local controller ensures smooth execution and adherence to local costmap constraints.

## 12. Obstacle Avoidance
Obstacle avoidance is fully operational. Dynamic and static obstacles not present on the global map are detected by the GPU LiDAR (`/scan`). Nav2 marks and clears these obstacles on the local costmap. Furthermore, a configured `CollisionMonitor` ensures the robot halts or slows down if collisions are imminent within the defined `FootprintApproach` polygon.

## 13. RTF Performance Optimization
Performance testing in the `small_house` world revealed severe physics bottlenecks.
- **Baseline RTF**: ~0.207 to 0.276
- **Final Config RTF**: ~1.0

**Explanation**: 29 non-interactive decorative environment models (such as furniture, boxes, portraits) defaulted to dynamic physics. Gazebo computed continuous mesh-to-mesh collision and gravity constraints for them against the floor. By injecting `<static>true</static>` into these SDF files, their physics overhead was eliminated. Because their collision geometries were fully preserved (no meshes were deleted), LiDAR and Nav2 obstacle avoidance continue to function flawlessly.

## 14. Static Models
- **Number of modified model.sdf files**: 29
- **Final directory**: `~/case_study/small_house/models_static_test`
- The original `models` directory was preserved unchanged.
- Static models dramatically improve performance by entirely bypassing the physics step for stationary visual/collision structures.

## 15. Troubleshooting

**A. RViz: "No map received" / "Frame [map] does not exist"**
- *Solution*: The `final_launch.py` was failing to pass the map argument down to the localization module. This was permanently fixed by forcing `final_launch.py` to pass `map:=/home/reyan/case_study/room_map.yaml` into `localization_launch.py`.

**B. RTF test hanging**
- *Solution*: The original benchmarking script waited indefinitely while collecting stats from Gazebo if the simulation stalled. The robust test script now utilizes a hard timeout (`timeout 30`).

**C. Gazebo starting paused**
- *Solution*: Running `gz sim -s` directly can start the simulation paused depending on world settings. Use `gz sim -s -r` to automatically resume execution.

**D. Correct RTF topic**
- *Solution*: Monitor Real-Time Factor via `gz topic -e -t /world/default/stats`.

## 16. Useful Commands
| Purpose | Command |
|---------|---------|
| Source ROS | `source /opt/ros/jazzy/setup.bash` |
| Build workspace | `colcon build --symlink-install` |
| Source workspace | `source install/setup.bash` |
| Launch simulation | `ros2 launch car_bot final_launch.py map:=/home/reyan/case_study/room_map.yaml world:=/home/reyan/case_study/small_house/small_house.world use_sim_time:=true` |
| Teleop | `ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p stamped:=true` |
| Inspect nodes | `ros2 node list` |
| Inspect topics | `ros2 topic list` |
| Inspect RTF | `gz topic -e -t /world/default/stats \| grep -i real_time_factor` |
| Kill Gazebo | `killall -9 ruby gz; pkill -f gazebo` |
| Kill ROS Launch | Press `Ctrl+C` in launch terminal |
| Check /map | `ros2 topic echo /map --once` |

## 17. Shutdown / Restart
- **Shutdown**: Press `Ctrl+C` in the primary launch terminal. If Gazebo hangs in the background, forcibly terminate it with `killall -9 ruby gz` or `pkill -f gz`.
- **Restart**: Always re-source your workspace (`source install/setup.bash`) and re-export the `GZ_SIM_RESOURCE_PATH` before relaunching.

## 18. Final Configuration Summary

| Component | Final configuration |
|-----------|--------------------|
| OS | Ubuntu 24.04 |
| ROS | ROS 2 Jazzy |
| Gazebo | Gazebo Harmonic |
| Robot | Differential-drive `car_bot` |
| World | `~/case_study/small_house/small_house.world` |
| Map | `~/case_study/room_map.yaml` |
| Map server | `nav2_map_server` |
| Localization | `AMCL` |
| Navigation | `Nav2` |
| LiDAR | GPU LiDAR (`/scan`) |
| Controller | `ros2_control` (DiffDrive) |
| Static models | `models_static_test` (29 models modified) |
| RTF | ~1.0 |

## 19. Important Rules
- **Do not** regenerate `room_map` unless intentionally executing a new mapping experiment.
- **Do not** replace `models_static_test` with the original models for the final configuration.
- **Do not** modify Nav2, SLAM, or controller parameter files without documenting a new performance benchmark.
- **Preserve** the working final configuration exactly as it is currently committed.

## 20. Project Status
**FINAL WORKING CONFIGURATION**

- **Navigation**: WORKING
- **Obstacle avoidance**: WORKING
- **Map**: WORKING
- **Localization**: WORKING
- **RTF**: approximately 1.0

---

## 21. Docker Development Environment & Portability Guide

### 21.1 Overview & Architecture
The project provides a fully self-contained, portable Docker environment built on **Ubuntu 24.04** and **ROS 2 Jazzy Desktop**, integrating:
- **Gazebo Harmonic** (`ros_gz_sim`, `ros_gz_bridge`, `ros_gz_image`)
- **Nav2 & SLAM Toolbox** (`navigation2`, `nav2_bringup`, `slam_toolbox`)
- **ros2_control** (`controller_manager`, `diff_drive_controller`, `joint_state_broadcaster`, `gz_ros2_control`)
- **Voice Navigation Pipeline** (FFmpeg, PyAudio, SpeechRecognition with Google bn-BD STT)
- **Qwen LLM Interface** (PyTorch, Hugging Face Transformers, PEFT, bitsandbytes)
- **GUI Pass-through** for Gazebo GUI and RViz2 via X11 socket forwarding
- **Hardware Acceleration** for AMD/Intel/NVIDIA DRI graphics, with automatic CPU Mesa fallback
- **External Volume Mounting** to keep large LLM weights (~5.8 GB) outside the Docker image layers

### 21.2 File Roles
- `Dockerfile`: Multi-layer optimized build installing ROS 2 Jazzy, Nav2, SLAM, Gazebo Harmonic, audio, and Python dependencies.
- `docker-compose.yml`: Configures container mounts (`/workspace`, `/tmp/.X11-unix`, `/dev/dri`, `/dev/snd`, `~/.cache/huggingface`), host networking, display variables, and optional NVIDIA profiles.
- `.dockerignore`: Excludes host build caches (`build/`, `install/`, `log/`, `voice_env/`, `.venv/`, `.git/`, `.vscode/`, `*.zip`, `*.safetensors`) from blowing up the Docker build context.
- `entrypoint.sh`: Auto-sources `/opt/ros/jazzy/setup.bash`, sources `/workspace/install/setup.bash` (if built), sets `GZ_SIM_RESOURCE_PATH`, creates backward-compatibility symlinks, and executes commands.

---

### 21.3 Host Prerequisites (Any Ubuntu 24.04 Laptop)
Before running Docker on the host laptop:

1. **Install Docker Engine & Docker Compose Plugin**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y ca-certificates curl gnupg
   sudo install -m 0755 -d /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   sudo chmod a+r /etc/apt/keyrings/docker.gpg

   echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
     sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

   sudo apt-get update
   sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

2. **Add User to Docker Group** (avoid running Docker with sudo):
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

3. **Grant X11 Display Access for GUI (Gazebo & RViz2)**:
   ```bash
   xhost +local:root
   ```
   *(To persist across reboots, add `xhost +local:root > /dev/null 2>&1` to `~/.bashrc`)*.

---

### 21.4 Building and Running via Docker Compose

1. **Build the Docker Image**:
   ```bash
   cd ~/case_study
   docker compose build
   ```
   *Note: For an NVIDIA GPU laptop with NVIDIA Container Toolkit, build with CUDA PyTorch:*
   ```bash
   docker compose build --build-arg TORCH_DEVICE=cuda
   ```

2. **Start an Interactive Container Shell**:
   ```bash
   # Standard DRI / CPU fallback container
   docker compose run --rm robot_dev bash

   # Or NVIDIA GPU accelerated container (if nvidia-container-toolkit is installed)
   docker compose --profile nvidia run --rm robot_dev_nvidia bash
   ```

---

### 21.5 Building ROS 2 Packages Inside the Container
Once inside the `/workspace` container shell:
```bash
# 1. Check and install any missing rosdep dependencies
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# 2. Build the workspace
colcon build --symlink-install

# 3. Source the freshly built workspace overlay
source install/setup.bash
```

---

### 21.6 Launching Simulation & Navigation Inside Docker

#### Option A: Full GUI Simulation (Gazebo + AMCL + Nav2 + RViz2)
Ensure `xhost +local:root` was executed on the host, then inside the container run:
```bash
ros2 launch car_bot final_launch.py \
  map:=/workspace/room_map.yaml \
  world:=/workspace/small_house/small_house.world \
  use_sim_time:=true
```

#### Option B: Headless Simulation (Server-Only, No Display Required)
If running remotely or verifying without an active monitor:
```bash
ros2 launch car_bot gazebo_launch.py gui:=false
```

#### Option C: Teleoperation
Open a second terminal on the host, enter the running container, and run:
```bash
docker exec -it car_bot_dev_container bash
source install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p stamped:=true
```

---

### 21.7 Running Voice Navigation and Qwen LLM Inside Docker

1. **Audio Device Access**:
   `/dev/snd` is passed directly from host to container in `docker-compose.yml`. Verify recording devices inside the container:
   ```bash
   arecord -l
   ```

2. **Run Bangla Voice Command Node**:
   ```bash
   ros2 run location_navigation voice_command_node
   ```

3. **Run LLM Navigation Action Dispatcher**:
   ```bash
   ros2 run location_navigation llm_to_nav '{"intent":"MOVE_TO_TARGET","target":"table"}'
   ```

4. **Run Regression & Verification Test Suite**:
   ```bash
   python3 /workspace/llm/QWEN_LINUX_TRANSFER/test_regression.py
   ```

---

### 21.8 VS Code Live Development Workflow
The workspace directory is mounted live at `/workspace`:
- Source code in `src/` can be edited directly on the host using **VS Code**, Sublime, or any IDE.
- Changes are immediately reflected inside the container without rebuilding the Docker image.
- To rebuild modified packages inside the container:
  ```bash
  colcon build --packages-select <package_name> --symlink-install
  source install/setup.bash
  ```

---

### 21.9 Adding New ROS 2 Packages
To add a new ROS 2 package:
1. Create or clone the package into `src/` on the host:
   ```bash
   cd ~/case_study/src
   ros2 pkg create --build-type ament_python my_new_pkg
   ```
2. Enter the container and install dependencies:
   ```bash
   docker exec -it car_bot_dev_container bash
   rosdep install --from-paths src --ignore-src -r -y
   ```
3. Build and test:
   ```bash
   colcon build --packages-select my_new_pkg --symlink-install
   source install/setup.bash
   ```

---

### 21.10 GPU Acceleration and CPU Fallback

| Mode | Configuration | Command |
|------|---------------|---------|
| **AMD / Intel / Mesa GPU** | Uses `/dev/dri` passthrough | `docker compose run --rm robot_dev bash` |
| **NVIDIA GPU** | Uses `nvidia-container-toolkit` | `docker compose --profile nvidia run --rm robot_dev_nvidia bash` |
| **CPU Fallback** | Software rasterizer (no GPU needed) | `LIBGL_ALWAYS_SOFTWARE=1 docker compose run --rm robot_dev bash` |

---

### 21.11 Transfer Workflow to Another Laptop (Offline / USB / LAN)

To transfer the entire environment to another Ubuntu 24.04 laptop:

#### On Source Laptop (Current Machine):
1. **Save Docker Image to Compressed Archive**:
   ```bash
   docker save car_bot_dev:latest | gzip > ~/car_bot_dev_image.tar.gz
   ```

2. **Bundle the Workspace (Excluding Build/Install Caches)**:
   ```bash
   cd ~
   tar --exclude='case_study/build' \
       --exclude='case_study/install' \
       --exclude='case_study/log' \
       --exclude='case_study/voice_env' \
       --exclude='case_study/llm/QWEN_LINUX_TRANSFER/.venv' \
       -czvf ~/case_study_workspace.tar.gz case_study/
   ```

3. **Copy to External USB Drive or SCP**:
   ```bash
   # Example via USB drive:
   cp ~/car_bot_dev_image.tar.gz /media/$USER/USB_DRIVE/
   cp ~/case_study_workspace.tar.gz /media/$USER/USB_DRIVE/
   ```

#### On Destination Laptop (New Machine):
1. **Install Docker Engine on Destination** (refer to Section 21.3).
2. **Copy Files from USB Drive to Home Directory**:
   ```bash
   cp /media/$USER/USB_DRIVE/car_bot_dev_image.tar.gz ~/
   cp /media/$USER/USB_DRIVE/case_study_workspace.tar.gz ~/
   ```
3. **Extract the Workspace**:
   ```bash
   cd ~
   tar -xzvf case_study_workspace.tar.gz
   cd ~/case_study
   ```
4. **Load the Docker Image**:
   ```bash
   docker load < ~/car_bot_dev_image.tar.gz
   ```
5. **Grant X11 GUI Access & Run**:
   ```bash
   xhost +local:root
   docker compose run --rm robot_dev bash
   ```
6. **Build and Launch**:
   ```bash
   colcon build --symlink-install
   source install/setup.bash
   ros2 launch car_bot final_launch.py \
     map:=/workspace/room_map.yaml \
     world:=/workspace/small_house/small_house.world \
     use_sim_time:=true
   ```
