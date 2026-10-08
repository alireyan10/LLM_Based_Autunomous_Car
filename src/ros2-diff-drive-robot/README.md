# LLM-Based Autonomous Car

A ROS 2 Jazzy and Gazebo Harmonic based autonomous differential-drive mobile robot project.

This project implements a complete mobile robot simulation pipeline including robot modeling, differential-drive control, LiDAR and camera sensing, SLAM mapping, localization, autonomous navigation, obstacle avoidance, and an LLM-based navigation foundation.

---

# 1. Project Overview

The main objective of this project is to develop and simulate an autonomous mobile robot capable of:

- Moving using differential-drive control
- Operating in Gazebo Harmonic
- Using a 2D LiDAR sensor for environment perception
- Using a camera sensor
- Visualizing robot data in RViz2
- Creating 2D maps using SLAM Toolbox
- Localizing itself using AMCL
- Planning and following paths using Nav2
- Avoiding obstacles autonomously
- Using the MPPI navigation controller
- Using Twist Mux for velocity arbitration
- Receiving keyboard teleoperation commands
- Providing a foundation for LLM-based natural-language navigation

---

# 2. Technologies Used

- Ubuntu 24.04 LTS
- ROS 2 Jazzy
- Gazebo Harmonic
- RViz2
- Nav2
- SLAM Toolbox
- ros2_control
- Differential Drive Controller
- MPPI Controller
- Twist Mux
- LiDAR
- Camera
- Python
- Git / GitHub
- Qwen LLM (navigation integration)

---

# 3. Main Features

## Robot Simulation

- Differential-drive mobile robot
- Wheel control using ros2_control
- Gazebo Harmonic simulation
- Robot sensors integration

## Sensors

- 2D LiDAR
- Camera

## Mapping

- SLAM Toolbox
- Teleoperation-based mapping
- Saved 2D map

## Localization

- AMCL
- Initial pose estimation using RViz2

## Navigation

- ROS 2 Nav2
- MPPI Controller
- Global and local costmaps
- Autonomous path following
- Obstacle avoidance

## Performance Optimization

The small-house environment contains many static objects.

To improve Gazebo simulation performance, optimized copies of the environment models were created with static objects.

The optimized models are stored in:

```text
small_house/models_static_test/


## Clone the project:
git clone https://github.com/alireyan10/LLM_Based_Autunomous_Car.git

## Go into the project:
cd LLM_Based_Autunomous_Car

## Required Dependencies

## Install required development tools:
sudo apt install python3-colcon-common-extensions python3-rosdep git

## Update rosdep:
rosdep update

## Install dependencies:

cd ~/case_study

rosdep install --from-paths src --ignore-src -r -y

#Build the Project

colcon build
source install/setup.bash

## For development with symlinks:
colcon build --symlink-install

## The project contains different launch files for different tasks.

gazebo_launch.py  ->	Launch robot in basic/empty Gazebo environment
slam_launch.py	  ->    Launch robot + Gazebo + SLAM Toolbox for mapping
final_launch.py	  ->    Launch small-house environment + localization + Nav2 autonomous navigation

## Teleoperate the Robot

ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p stamped:=true

## Small House Environment

small_house/models_static_test  ->it contains the optimized static environment models.

## Launch Small House World

export GZ_SIM_RESOURCE_PATH=$HOME/case_study/small_house/models_static_test:$HOME/case_study/small_house:$HOME/case_study/small_house/maps

gz sim ~/case_study/small_house/small_house.world

## Small House Autonomous Navigation

export GZ_SIM_RESOURCE_PATH=$HOME/case_study/small_house/models_static_test:$HOME/case_study/small_house:$HOME/case_study/small_house/maps

ros2 launch car_bot final_launch.py \
map:=/home/reyan/case_study/room_map.yaml \
world:=/home/reyan/case_study/small_house/small_house.world \
use_sim_time:=true



