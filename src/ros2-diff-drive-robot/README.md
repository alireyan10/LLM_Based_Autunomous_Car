# ROS 2 Differential Drive Robot Simulation

A complete differential-drive mobile robot simulation built with ROS 2 Jazzy, Gazebo Harmonic, and Nav2. The project demonstrates robot simulation, mapping, localization, autonomous navigation, and ROS 2 control using modern robotics tools.

## Features

- Differential drive controller using ros2_control
- Gazebo Harmonic simulation
- Camera sensor integration
- 2D LiDAR integration
- RViz visualization
- SLAM mapping with SLAM Toolbox
- AMCL localization
- Autonomous navigation with Nav2
- MPPI Controller
- Twist Mux velocity arbitration
- Keyboard teleoperation

## Requirements

- Ubuntu 24.04
- ROS 2 Jazzy
- Gazebo Harmonic
- git

make sure to source the setup file on every terminal
```bash
source /opt/ros/jazzy/setup.bash
```

## Installation 

update packages
```bash
sudo apt update
sudo apt upgrade
```

Install Dependencies
```bash
sudo apt install python3-colcon-common-extensions python3-rosdep git
```

Update Rosdep
```bash
rosdep update
```


In your workspace/src folder clone this repository
```bash
git clone https://github.com/Safin-Sadman/ros2-diff-drive-robot.git
```

Go back to the root of your workspace and install all ROS dependencies
```bash
rosdep install --from-paths src --ignore-src -r -y
```

Then build the project
```bash
colcon build
```

## Running the project

In the root of your workspace run
```bash
source install/setup.bash
```

Run the bot with gazebo simulation
```bash
ros2 launch car_bot gazebo_launch.py
```
![Gazebo Simulation](images/gazebo.png)

You can now visualize camera and lidar data in rviz

![RViz](images/rviz.png)

To Control the bot , lets run teleop command in a new terminal
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p stamped:=true
```
Now you can control your car with keyboard.
press ctrl-c to exit.

## Mapping

Run gazebo sim with slam toolbox for mapping
```bash
ros2 launch car_bot slam_launch.py
```
Now run teleop_twist_key to move around and create map
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p stamped:=true
```
After creating map, you can save it in slam_toolbox plugin in car_bot/map folder for future navigation, for this project I have already saved a map.
press ctrl-c to exit.

## Localization & Navigation

Run gazebo sim with nav2
```bash
ros2 launch car_bot final_launch.py
```
After loading the map set initial position and pose on rviz with "2D pose estimate" button for localization, this will set the initial pose of the robot.
![Localization Demo](gif/localization.gif)

Then wait for some time to load the costmap. after that click "2D goal pose" to navigate autonomously.
![Navigation Demo](gif/navigation.gif)



## Acknowledgements

This project was inspired by:

Original project:
https://github.com/joshnewans/my_bot

ideas and references were taken from the original project.
The original project is licensed under Apache License 2.0.