# Optimizations and pipeline timings

Everything here is measured, not estimated. Test rig unless stated otherwise:

- **GPU** RTX 3050 6 GB laptop, compute 8.6, **FP32** (FP16 is `__aarch64__`-gated)
- **Bag** `2011_09_29_drive_0004_sync_bag`, ~124 000 points/cloud, 19 663 pillars
- **Method** both pipelines as real ROS nodes, fed lock-step, wall clock inside
  the subscriber callback, `PP_TIME_CB=1`
- **Build** `-DCMAKE_BUILD_TYPE=Release`

---

## 1. Headline

| | OLD (PyTorch node) | NEW (TensorRT node) |
|---|---:|---:|
| callback, median | **135.9 ms** | **20.80 ms** |
| throughput | **7.4 Hz** | **48.1 Hz** |
| speedup | | **6.5×** |

Reproduce OLD with `tools/run_old_pipeline.py` — it wraps their real
`LidarObjectDetectorNode`, so this is their compiled CUDA `voxel_op`, their
PointPillars forward, their Python AB3DMOT, their message building.

## 2. Where the 20.8 ms goes

| stage | ms | notes |
|---|---:|---|
| repack 124k points (CPU) | 0.30 | field-offset lookup by name, single pass |
| `generateVoxels` (CUDA) | 0.11 | |
| `generateFeatures` (CUDA) | 0.21 | |
| **`doinfer` (TensorRT)** | **19.22** | **92% of the frame — GPU-bound** |
| decode + NMS + track + publish | ~0.9 | |

Stage figures come from a `-DPERFORMANCE_LOG=1` build; the 20.8 ms total from a
default build. **The pipeline is GPU-bound.** No amount of host-side work
matters until the engine itself gets faster — FP16 on Orin is the lever, not CPU
tuning.

Spread is tight: p10 20.60 ms, median 20.80 ms, p90 21.20 ms over 363 frames.

---

## 3. Optimizations, with measured effect

### 3.1 Build the TensorRT engine once, not per frame — **the big one**

The inherited code constructed `PointPillar` inside the subscriber callback, so
every cloud rebuilt or re-deserialised the engine. It is now built once in the
node constructor, and the serialised engine is cached next to the `.onnx`, keyed
on model hash + TRT version + GPU arch + precision.

**Effect:** first run pays ~1–2 min to build; every run after that starts in
seconds, and per-frame cost drops from "rebuild the engine" to 20.8 ms.

### 3.2 `${GENCODE}` never reached nvcc

`CMakeLists.txt` computed the correct `-gencode arch=compute_86,...` string and
then never passed it to the compiler, so the CUDA kernels were built for the
default architecture.

```cmake
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    ${GENCODE}                      # <- was computed, then dropped
    -ccbin ${CMAKE_CXX_COMPILER} ...)
```

Set `GPU_SMS` to match the target: **86** for RTX 30-series, **87** for Orin.

### 3.3 Per-stage timing instrumentation off by default

`PERFORMANCE_LOG` inserts `cudaDeviceSynchronize()` / `cudaEventSynchronize()`
plus a device-to-host `cudaMemcpy` between stages, and prints three lines per
frame.

| build | median callback |
|---|---:|
| `PERFORMANCE_LOG=0` (default) | **20.80 ms** |
| `PERFORMANCE_LOG=1` | 21.00 ms |

**Effect: ~1%.** An earlier comment in `pointpillar.h` claimed it cost "roughly
half the frame rate" — that was wrong, and the measurement above is why it now
says otherwise. The extra syncs buy little because the callback is already
serialised around one blocking TensorRT call. It is off because the log spam is
noise, not because it is expensive.

Turn it back on when profiling:

```bash
colcon build --packages-select cuda_pointpillars_ros \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DPERFORMANCE_LOG=1
```

### 3.4 Detection and tracking in one process

The old layout ran a Python tracker inside the detector node. The new one runs a
C++ AB3DMOT in the same process as the TensorRT detector — boxes go straight
from the decode into the tracker with no serialisation, no IPC, no second node.

**Effect:** tracking + message building costs ~0.9 ms of the 20.8 ms frame.
Runtime dependencies drop from torch + numpy + scipy + filterpy + numba +
ros2_numpy to **Eigen only**.

### 3.5 Candidate caps (`nms_pre`, `max_num`)

Added for *correctness* (see §4), but they also cut NMS work: the decode now
sorts and suppresses ~100 anchors' worth of boxes instead of ~350.

### 3.6 Device buffer reuse

The point buffer grows only when a cloud needs more room than the largest seen
so far, instead of allocating per frame. Anchor grouping in the decode sorts a
few hundred indices rather than indexing a 321 408-entry table per frame.

---

## 4. Correctness fixes found by measurement

These are not speed work, but they are the substance of the port. Full detail in
`docs/COMPARISON.md`.

| # | defect | effect before fix |
|---|---|---|
| 1 | `nms_pre` / `max_num` caps missing | 44.4 boxes/frame vs reference 19.9 |
| 2 | argmax-only class emission | lost ~6% of reference detections |
| 3 | anchor grid endpoint-to-endpoint, not bin centres | systematic 0.08 m offset on every box |
| 4 | z origin: centre emitted, bottom expected | every box rendered h/2 too high |
| 5 | NMS suppressed across classes | a Car could delete an overlapping Pedestrian |
| 6 | pillar buffer written out of bounds | memory corruption above `MAX_VOXELS` |
| 7 | box count read from a racy slot | last writer won, not the largest |

Result after fixing all seven:

| metric | before | after |
|---|---:|---:|
| objects/frame (reference: 42.8) | 83.1 | **45.0** |
| reference detections reproduced within 0.5 m | 72.3% | **94.0%** |
| median 3D centre error | 0.131 m | **0.0001 m** |
| class agreement | 98.6% | **99.8%** |

---

## 5. Measurement traps hit along the way

Recorded because each one produced a confidently wrong number first.

- **Do not benchmark through `tools/play_bag.py`.** It spends ~230 ms/frame
  decompressing 4 MB MCAP clouds in Python. An early "TensorRT is slower than
  PyTorch" conclusion came from comparing that harness against in-process
  PyTorch. Time the node's callback, not the round trip.
- **Check the topic is quiet before timing.** One comparison run was taken while
  a second bag replay was publishing to `/kitti/velo`; it reported OLD at 77.6 ms
  instead of the correct 135.9 ms.
- **Input QoS is `RELIABLE / KEEP_LAST / depth=1`.** A free-running publisher
  silently drops frames whenever inference is slower than the publish rate. Use
  `--lockstep` for any correctness run.
- **Tensor-level agreement is not end-to-end agreement.** `cls_preds`
  correlation was 0.999912 while the decode still had four defects.

---

## 6. Not done yet

- **FP16.** Gated to `__aarch64__`, so x86 runs FP32. On Orin this should be the
  single largest remaining win, since 92% of the frame is inference.
- **INT8.** Not attempted; would need a calibration set.
- **Livox model.** `config/livox.yaml` and `docs/TRAINING_HANDOFF.md` are ready;
  the model itself is with the training owner.
