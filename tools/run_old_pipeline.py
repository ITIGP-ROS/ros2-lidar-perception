#!/usr/bin/env python3
"""Run the ORIGINAL Python detector node, for an apples-to-apples comparison.

This instantiates the real `LidarObjectDetectorNode` from
reference/lidar_object_detection_py -- its own voxelisation (compiled voxel_op),
its own PointPillars forward, its own AB3DMOT, its own message building. The
only addition is a wall-clock timer around the callback, matching PP_TIME_CB on
the C++ node so the two numbers mean the same thing.

    python3 tools/run_old_pipeline.py --weights <path/to/epoch_160.pth>
"""
import argparse
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REF = REPO / 'reference' / 'lidar_object_detection_py'
for p in (str(REF), str(REPO / 'reference' / 'ros2_numpy')):
    if p not in sys.path:
        sys.path.insert(0, p)

import rclpy  # noqa: E402
import torch  # noqa: E402
from lidar_object_detection.lidar_object_detector_node import LidarObjectDetectorNode  # noqa: E402


class Timed(LidarObjectDetectorNode):
    def __init__(self, weights):
        super().__init__(weights)
        self.times = []
        # the original logs a line per frame; silence it so it does not distort
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.WARN)

    def lidar_point_cloud_callback(self, msg):
        t0 = time.perf_counter()
        super().lidar_point_cloud_callback(msg)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        self.times.append(dt)
        n = len(self.times)
        if n % 5 == 0:
            print(f"  [{n:3d}] callback {dt:7.1f} ms", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=str(REF / 'weights' / 'epoch_160.pth'))
    ap.add_argument('--frames', type=int, default=40)
    a = ap.parse_args()

    rclpy.init()
    node = Timed(a.weights)
    print("old pipeline ready; waiting for /kitti/velo", flush=True)
    idle = 0
    try:
        while rclpy.ok() and len(node.times) < a.frames and idle < 120:
            before = len(node.times)
            rclpy.spin_once(node, timeout_sec=0.5)
            idle = idle + 1 if len(node.times) == before else 0
    except KeyboardInterrupt:
        pass
    finally:
        t = node.times[2:]     # drop warm-up
        if t:
            print(f"\nOLD PIPELINE  n={len(t)}  median={st.median(t):.1f} ms  "
                  f"mean={st.mean(t):.1f} ms  -> {1000/st.median(t):.1f} Hz")
        rclpy.shutdown()


if __name__ == '__main__':
    main()
