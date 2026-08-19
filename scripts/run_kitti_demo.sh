#!/usr/bin/env bash
# End-to-end KITTI demo that works WITHOUT ros2 launch / ros2 bag / ros2 run,
# none of which are installed on this workstation. Starts the pipeline, replays
# the bag, and prints the tracked objects as they come back.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BAG="${BAG:-$HOME/Downloads/KITTI_Dataset/newest/2011_09_29_drive_0004_sync_bag}"
MODEL="${MODEL:-$HERE/src/cuda_pointpillars_ros/model/pointpillar_gp.onnx}"
FRAMES="${FRAMES:-40}"

source /opt/ros/humble/setup.bash
source "$HERE/install/setup.bash"

[ -f "$MODEL" ] || { echo "model not found: $MODEL"; echo "run tools/export_pytorch_to_onnx.py first"; exit 1; }
[ -d "$BAG" ]   || { echo "bag not found: $BAG"; exit 1; }

echo "=== starting pipeline (first run builds the TensorRT engine, ~1-2 min) ==="
"$HERE/install/cuda_pointpillars_ros/lib/cuda_pointpillars_ros/pc_process" --ros-args \
  --params-file "$HERE/install/lidar_perception_bringup/share/lidar_perception_bringup/params/kitti.yaml" \
  -p model_path:="$MODEL" &
NODE=$!
trap 'kill $NODE 2>/dev/null || true' EXIT

for i in $(seq 1 60); do
  sleep 3
  kill -0 $NODE 2>/dev/null || { echo "node exited -- see the error above"; exit 1; }
  [ $i -ge 8 ] && break
done

echo
echo "=== replaying $FRAMES frames (lock-step, so no frames are dropped) ==="
python3 "$HERE/tools/play_bag.py" --bag "$BAG" --topic /kitti/velo \
        --count "$FRAMES" --lockstep --watch object_detections_3d
