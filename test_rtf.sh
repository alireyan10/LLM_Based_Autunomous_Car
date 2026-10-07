#!/bin/bash

# Cleanup trap to ensure Gazebo is killed on exit
cleanup() {
    echo "Cleaning up Gazebo process..."
    if [ -n "$GZ_PID" ]; then
        kill $GZ_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting Gazebo server-only with static models..."
export GZ_SIM_RESOURCE_PATH=/home/reyan/case_study/small_house/models_static_test:$GZ_SIM_RESOURCE_PATH

# Launch Gazebo in the background
gz sim -s /home/reyan/case_study/small_house/small_house.world &
GZ_PID=$!

echo "Waiting 15 seconds for simulation to stabilize..."
sleep 15

echo "Collecting up to 10 RTF samples (30 second timeout)..."
# Use timeout to enforce 30 seconds limit on the gz topic command
RAW_SAMPLES=$(timeout 30 gz topic -e -t /world/default/stats | grep -m 10 -i real_time_factor | awk '{print $2}')

echo "---"
echo "COLLECTED SAMPLES:"

TOTAL=0
COUNT=0
# Read and validate numeric samples
for val in $RAW_SAMPLES; do
    # Remove any stray carriage returns or quotes just in case
    val=$(echo "$val" | tr -d '\r"')
    
    # Check if the value is a valid numeric float/integer
    if [[ $val =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "Sample $((COUNT+1)): $val"
        TOTAL=$(echo "$TOTAL + $val" | bc -l)
        COUNT=$((COUNT + 1))
    fi
done

if [ $COUNT -eq 0 ]; then
    echo "Error: Zero valid RTF samples collected."
    exit 1
fi

AVG_RTF=$(echo "$TOTAL / $COUNT" | bc -l)
BASELINE=0.276

# Calculate percentage improvement
IMPROVEMENT=$(echo "scale=2; (($AVG_RTF - $BASELINE) / $BASELINE) * 100" | bc -l)

echo ""
echo "==== CONCISE REPORT ===="
echo "1. What was changed: Created a test copy of models and added <static>true</static> to all 29 dynamic environment models."
echo "2. Number of models changed: 29"
echo "3. Baseline RTF: 0.276"
echo "4. Test RTF: $(printf "%.3f" $AVG_RTF)"
echo "5. Improvement %: ${IMPROVEMENT}%"
echo "6. Whether static models are actually beneficial: (Awaiting user decision based on results)"
echo "7. Recommended next optimization: (If insufficient, investigate LiDAR ray intersections with complex .DAE collision meshes)"
echo "========================"
