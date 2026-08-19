# Running the pipeline

Detection **and** tracking run in one process. Unlike the old pipeline there is
no separate tracker node: the TensorRT detector hands boxes straight to the C++
AB3DMOT tracker, and the node publishes `object_detection_msgs/Object3dArray`
with `track_id` already filled in.

```
/kitti/velo ──▶ cuda_pointpillars_node ──▶ /object_detections_3d ──▶ IVI
              (TensorRT + AB3DMOT)                │
                                                  └──▶ object3d_visualizer_node
                                                       └▶ /object_detection_visualization  (RViz)
```

---

## Heads-up: the ros2 CLI on this workstation is incomplete

`ros2` here provides only `control / daemon / node / param / service`. **`ros2
launch`, `ros2 run`, `ros2 bag` and `ros2 topic` are NOT installed.** Section 3
onwards assumes them; if you have not installed them yet, use the one-command
path below, which needs none of them.

```bash
sudo apt install ros-humble-ros2launch ros-humble-ros2run \
                 ros-humble-ros2bag ros-humble-ros2topic \
                 ros-humble-rosbag2-storage-mcap
```

### One-command demo (works today, no extra packages)

```bash
cd ~/ros2_ws_gp/src/ros2-lidar-perception
FRAMES=40 ./scripts/run_kitti_demo.sh
```

It starts the pipeline, replays the bag lock-step so no frames are dropped, and
prints the tracked objects. Verified output:

```
[   9] n=11  30:Ped(19.2,-13.1)s0.37 29:Ped(7.0,27.5)s0.43 28:Ped(8.4,20.0)s0.44 ...
[  12] n= 7  36:Ped(14.9,-18.0)s0.38 35:Ped(16.2,-18.1)s0.38 30:Ped(19.2,-13.1)s0.37 ...
SUMMARY: 12 published, 12 answered, 28 distinct track ids
```

Ids persisting across frames (30 over frames 9-12, 26 over 8-12) is the signal
that detection *and* tracking are both working. Override `BAG=`, `MODEL=` or
`FRAMES=` as needed.

---

## 1. Build

```bash
cd ~/ros2_ws_gp/src/ros2-lidar-perception
source /opt/ros/humble/setup.bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DGPU_SMS=86
source install/setup.bash
```

`GPU_SMS` must match the GPU: **86** for an RTX 30-series laptop, **87** for
Jetson Orin. Getting this wrong is not fatal but costs startup time and speed.

## 2. Generate the ONNX from the PyTorch weights

Only needed once, or whenever the weights change.

```bash
python3 tools/export_pytorch_to_onnx.py \
    --weights reference/lidar_object_detection_py/weights/epoch_160.pth \
    --output  src/cuda_pointpillars_ros/model/pointpillar_gp.onnx
```

Expect `voxels [40000, 32, 9]` and `PPScatterPlugin present: True`.

## 3. Launch

The model path and the visualiser threshold now live in
`src/lidar_perception_bringup/params/<profile>.yaml`, so the usual case needs no
arguments at all:

```bash
ros2 launch lidar_perception_bringup perception.launch.py
```

Any argument given on the launch line overrides the file; omitting one leaves
the file's value alone:

```bash
ros2 launch lidar_perception_bringup perception.launch.py \
    model_path:=$PWD/src/cuda_pointpillars_ros/model/pointpillar_gp.onnx \
    confidence_threshold:=0.5
```

`model_path` must be an absolute path to a file that exists — the node checks
and says so plainly rather than letting the ONNX parser fail deep in its own log.

### Controlling what reaches `/object_detections_3d`

There are two different thresholds and it matters which one you reach for:

```bash
ros2 launch lidar_perception_bringup perception.launch.py score_threshold:=0.6
```

`score_threshold` filters detections **before tracking**, so weak boxes never
reach the tracker and cannot spawn tracks. Measured on the KITTI bag:

| `score_threshold` | objects/frame | lowest published score | track ids over 20 frames |
|---|---:|---:|---:|
| `0.0` (off) | 10.35 | 0.351 | 87 |
| `0.6` | 4.35 | 0.601 | 12 |

It can only ever **raise** the compiled `SCORE_THRESH` — anything below that was
discarded on the GPU and no longer exists. Ask for less and the node says so
rather than silently doing nothing. To genuinely go lower, change `SCORE_THRESH`
in `config/*.yaml`, regenerate `params.h` and rebuild.

`confidence_threshold` is a different knob: it only hides boxes in the
visualiser, and does not change the published topic at all.

Wait for:

```
[INFO] engine matches params.h (3 classes, 6 anchors, grid 432x496, head 216x248, max 40000 pillars)
[INFO] model ready
```

**The first run builds the TensorRT engine and takes ~1-2 minutes.** It is cached
next to the .onnx (keyed on model hash + TRT version + GPU arch + precision), so
later runs start instantly.

If instead you see `The loaded engine does not match params.h`, the model and
the runtime config disagree — the message names exactly which tensor. Regenerate
`params.h` from the config that matches your model:

```bash
python3 tools/gen_params_h.py config/gp_pytorch_kitti.yaml \
    -o src/cuda_pointpillars_ros/include/params.h
colcon build --packages-select cuda_pointpillars_ros --cmake-args -DGPU_SMS=86
```

## 4. Play the bag

In a second terminal:

```bash
source /opt/ros/humble/setup.bash
ros2 bag play ~/Downloads/KITTI_Dataset/newest/2011_09_29_drive_0004_sync_bag --rate 0.3
```

Without `ros2 bag`, use the bundled player (same thing, reads MCAP directly):

```bash
python3 tools/play_bag.py --bag ~/Downloads/KITTI_Dataset/newest/2011_09_29_drive_0004_sync_bag \
        --topic /kitti/velo --rate 3 --watch object_detections_3d
```

Add `--lockstep` to publish only after each reply — no dropped frames, at the
cost of running at inference speed rather than real time.

Real-time playback is fine. **The node processes a frame in 22 ms (~45 Hz)** on
an RTX 3050 at FP32, comfortably ahead of the bag's 10 Hz — see the performance
section below. Slow the rate only if you want to watch it in RViz.

No topic remap is needed — `input_topic` is a parameter and the KITTI profile
already points at `/kitti/velo`.

## 5. Look at it

```bash
rviz2
```

- **Fixed Frame:** `velo_link`
- add **PointCloud2** on `/kitti/velo`
- add **MarkerArray** on `/object_detection_visualization`

Or check the topic directly:

```bash
ros2 topic hz  /object_detections_3d
ros2 topic echo /object_detections_3d --field objects[0].track_id
```

## What "working" looks like

On this bag, roughly:

- **5-13 objects per frame** (not ~80 — that means `score_thresh` is too low)
- **scores 0.35-0.67**
- **track ids persist across frames.** Most tracks live 4+ frames; the clearest
  example is a pedestrian crossing at constant `y ≈ -13.0` held under one id for
  9 consecutive frames while moving 12.5 m.
- mostly **Pedestrian**. That is the model's genuine behaviour on this data, not
  a bug — the PyTorch reference peaks at 0.49 for Pedestrian versus 0.30 for
  Car.

---

## Switching to Livox

```bash
ros2 launch lidar_perception_bringup perception.launch.py \
    profile:=livox model_path:=/path/to/livox_model.onnx
```

Before that model exists, three things must line up — see
`docs/TRAINING_HANDOFF.md`:

1. `config/livox.yaml` must match the OpenPCDet training config
2. `params.h` regenerated from it with `tools/gen_params_h.py`
3. `intensity_scale` must match how training normalised intensity
   (`255.0` for a raw Livox reflectivity channel, `1.0` for KITTI)

The startup check will refuse to run on a mismatch rather than emitting noise.

---

## Knobs

| where | what | note |
|---|---|---|
| `params/*.yaml` → `model_path` | which .onnx to load | absolute path; override with `model_path:=` |
| `params/*.yaml` → `score_threshold` | cut-off on what is **published** to `/object_detections_3d` | override with `score_threshold:=`; can only raise `SCORE_THRESH`, never lower it |
| `params/*.yaml` → `confidence_threshold` | visualiser display cut-off only | does **not** affect what is published; override with `confidence_threshold:=` |
| `config/*.yaml` → `SCORE_THRESH` | the real detector threshold | compile-time; regenerate `params.h` and rebuild |
| `params/*.yaml` → `class_remap` | detector class id → message label | `[0,1,2]` for this model, `[2,0,1]` for KITTI-order models |
| `params/*.yaml` → `intensity_scale` | must match TRAINING, not the sensor | silent accuracy loss if wrong |
| `params/*.yaml` → `convert_center_z_to_bottom` | `true`; the decode emits a CENTRE z, the corners want a BOTTOM z | `false` renders every box h/2 too high |
| `config/*.yaml` → `ANCHOR_ALIGN_CENTER` | anchor grid: bin centres (this model) vs endpoint-to-endpoint (OpenPCDet/NVIDIA) | wrong choice offsets every box ~0.08 m |
| `config/*.yaml` → `NMS_PRE` / `MAX_NUM` | candidate caps, mirroring the PyTorch decode | omitting them doubles the box count |

## Debug env vars

```bash
PP_DEBUG_PILLARS=1   # per-frame pillar count + first pillars' points
PP_DUMP_CLS=<dir>    # dump raw cls/box/dir tensors for frame 0
```

`PP_DUMP_CLS` is how the pipeline was validated against PyTorch (correlation
0.9999 on all three outputs).


---

## Measured performance (RTX 3050, FP32, 124k-point KITTI clouds)

```
generateVoxels                      0.35 ms
generateFeatures                    0.27 ms
doinfer  (TensorRT)                20.25 ms
--------------------------------------------
GPU total                          20.87 ms

repack 124k points (CPU)             0.4 ms
full callback (repack -> publish)   22.0 ms   ->  ~45 Hz
```

The pipeline is **GPU-bound on inference** and runs ~4.5x faster than the bag's
10 Hz. On Jetson Orin, `kFP16` is enabled (it is gated to `__aarch64__`), so
inference should be faster still.

**Do not benchmark with `tools/play_bag.py`.** Callbacks arrive ~250 ms apart
under it while each takes 22 ms — the other ~230 ms is the Python MCAP reader
decompressing and deserialising a 4 MB cloud per frame. That is a property of
the test harness, not the pipeline; a real lidar driver publishes directly and
does not pay it.

Turn the per-stage numbers back on with `PERFORMANCE_LOG 1` in
`include/pointpillar.h` (it costs 3 full device syncs per frame, so leave it off
otherwise) and the callback breakdown with `PP_TIME_CB=1`.
