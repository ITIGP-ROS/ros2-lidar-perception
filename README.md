# ros2-lidar-perception

Integrated LiDAR perception pipeline: TensorRT PointPillars detection + AB3DMOT
multi-object tracking, in a single C++ process, targeting Jetson Orin NX.

Merged from two upstream repositories (history preserved via `git subtree`):

- `CUDA-PointPillars-ROS2` — https://github.com/Abdulrahman2200925/CUDA-PointPillars-ROS2 (branch `vis`)
- `ros2-lidar-object-detection` — https://github.com/ITIGP-ROS/ros2-lidar-object-detection (branch `master`)

The IVI-facing contract is `object_detection_msgs/Object3dArray` and is unchanged
from the second repo.
