#!/bin/bash
set -e

# Setup ROS 2 Jazzy environment
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source "/opt/ros/jazzy/setup.bash"
fi

# Setup local workspace overlay if built
if [ -f "/workspace/install/setup.bash" ]; then
    source "/workspace/install/setup.bash"
fi

# Configure Gazebo Harmonic resource path for small_house static models and worlds
export GZ_SIM_RESOURCE_PATH="/workspace/small_house/models_static_test:/workspace/small_house:/workspace/small_house/maps:${GZ_SIM_RESOURCE_PATH:-}"

# Create compatibility symlink for legacy hardcoded paths without modifying existing workspace files
if [ ! -d "/home/reyan/case_study" ]; then
    mkdir -p /home/reyan
    ln -sfn /workspace /home/reyan/case_study
fi

# Enable unbuffered python output
export PYTHONUNBUFFERED=1

# Execute provided command or default interactive bash
exec "$@"
