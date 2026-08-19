# reference/

Not built (see `COLCON_IGNORE`). This is the original Python implementation,
kept verbatim as the porting reference and as the source of the golden test
vectors in `src/lidar_tracking/test/data/`.

The authoritative pieces:

- `lidar_object_detection_py/lidar_object_detection/tracking/` — vendored AB3DMOT
  (MIT, xinshuoweng/AB3DMOT), adapted. Ported to C++ in `src/lidar_tracking`.
- `lidar_object_detection_py/lidar_object_detection/utils/process.py`
  `bbox3d2corners()` — the IVI-facing corner generation. This is the function
  that makes the poses correct; port it literally, do not re-derive it.

Do not "fix" the w/l convention disagreement between `tracking/box.py` and
`utils/process.py`. It is pre-existing behaviour in the working system, applied
identically to detections and tracks, and the tuned thresholds in
`tracking/model.py` were validated against it.
