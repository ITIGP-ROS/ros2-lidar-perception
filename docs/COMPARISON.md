# Old pipeline vs new — speed and equivalence

All figures measured on this workstation: **RTX 3050 6 GB laptop, FP32**, KITTI
bag `2011_09_29_drive_0004_sync_bag`, ~124 000 points/cloud.

Both pipelines run as **real ROS nodes**, fed the identical bag lock-step, timed
identically (wall clock inside the subscriber callback), same machine.

## Speed

| | OLD (PyTorch node) | NEW (TensorRT node) | ratio |
|---|---:|---:|---:|
| callback, median | **135.9 ms** | **20.8 ms** | |
| throughput | **7.4 Hz** | **48.1 Hz** | **6.5×** |

The OLD figure is the genuine article: their compiled CUDA `voxel_op`, their
PointPillars forward, their Python AB3DMOT, their message building. Reproduce
with `tools/run_old_pipeline.py` (build the ops first: `cd
reference/lidar_object_detection_py/lidar_object_detection/ops && python3
setup.py build_ext --inplace`).

Breakdown of the NEW node's ~21 ms:

```
repack 124k points (CPU)   0.3 ms
generateVoxels    (CUDA)   0.12 ms
generateFeatures  (CUDA)   0.21 ms
doinfer      (TensorRT)   19.2 ms    <- dominates; GPU-bound
decode + NMS + track + publish  ~1 ms
```

FP16 is gated to `__aarch64__`, so **x86 is FP32**; Orin should be faster still.

**Do not benchmark with `tools/play_bag.py`** — it adds ~230 ms/frame
decompressing 4 MB MCAP clouds in Python. That is the harness, not the node.

## Output equivalence

Same 33 frames, both nodes at `score_thr 0.1` (the reference's own default),
matching each OLD published object to its nearest NEW one in **3D**:

| metric | value |
|---|---:|
| OLD detections reproduced within 0.5 m | **94.0%** (1327/1411) |
| median 3D centre error | **0.0001 m** |
| 90th percentile 3D centre error | **0.0006 m** |
| median z offset (NEW − OLD) | **−0.0000 m** |
| class agreement on matches | **99.8%** |
| objects/frame | OLD 42.8, NEW 45.0 |
| distinct track ids over 33 frames | OLD 383, NEW 392 |

At the shipped operating point (`SCORE_THRESH 0.35`): OLD 8.0 objects/frame vs
NEW 8.5, **93.5% reproduced, median error 0.0002 m**.

A median centre error of 0.1 mm is float32 rounding. The decode is equivalent.

### What it took to get there

Four real defects, each found by measurement rather than inspection:

1. **`nms_pre` was missing.** The reference keeps only the **top 100 anchors**
   by max class score out of **321 408**, before decode or NMS
   (`pointpillars.py:263`). The CUDA kernel decoded every anchor above the score
   threshold — ~350 per frame. This alone produced 44.4 boxes/frame against the
   reference's 19.9. `max_num = 50` was also missing.
2. **Argmax-only class emission.** The reference emits a box for *every* class
   above threshold, so one anchor can yield two boxes; the kernel emitted only
   the argmax. Worth ~6% of the reference's detections.
3. **Anchor grid placement.** The kernel spanned endpoint-to-endpoint
   (`min + i*span/(n-1)`, OpenPCDet's default) while this repo's model was
   trained on bin centres (`min + (i+0.5)*span/n`). A systematic 0.08 m mean
   offset — this was the entire residual 0.131 m error. Now selectable via
   `ANCHOR_ALIGN_CENTER`, since the two lineages genuinely differ.
4. **Z origin.** The decode emits a box **centre** z; `bbox3d2corners` treats z
   as the **bottom**. The reference converts inside `anchors2bboxes`
   (`z = z - h / 2`), one layer below where the decode loop is. Without it every
   box renders exactly h/2 too high — visible in RViz as boxes floating above
   their objects. `convert_center_z_to_bottom` now defaults to **true**.

The count divergence was NOT an NMS problem, which is where I looked first and
where the earlier version of this document pointed. Two hypotheses died on
measurement: that the extra boxes were NMS-suppressible duplicates (they were
spatially distinct — 1.9% self-proximity vs OLD's 1.6%), and that the NMS
implementations differed materially (both use rotated-IoU; the C++ needed only
to stop suppressing across classes).

### Remaining differences, all accounted for

- **Cyclist tracks.** The Python has `thres = +2.0` for Cyclist, a sign error
  that rejects every association. The C++ ships the corrected `-2.0`. Measured:
  reverting to `+2.0` drops NEW's unmatched Cyclists from **53 to 3** and
  objects/frame from 45.0 to 43.1 (OLD: 42.8). This is the intended divergence,
  not a decode error.
- **~6% unmatched Pedestrians**, median score **0.22** — below the shipped 0.35
  threshold. Marginal track births/deaths, expected of any tracker
  reimplementation with different association tie-breaking. Cars: **21 of 21
  reproduced**.

## Structural differences

| | old | new |
|---|---|---|
| processes | detector node + visualiser | **same two** |
| tracking | Python AB3DMOT, inside the detector node | **C++ AB3DMOT, in the same process** |
| inference | PyTorch eager, `epoch_160.pth` | TensorRT engine from ONNX |
| voxelisation | compiled `voxel_op` extension | CUDA kernels in-pipeline |
| runtime deps | torch, numpy, scipy, filterpy, numba, ros2_numpy | **Eigen only** |
| detector threshold | Python constant | `SCORE_THRESH` in `config/*.yaml` → `params.h` |
| model path / display threshold | launch args | **`params/*.yaml`**, overridable on the launch line |
| config source of truth | scattered | one YAML → generates `params.h` |
| model/config mismatch | silent wrong output | **node refuses to start**, naming the tensor |

**Class ids are unchanged** (`PEDESTRIAN=0, CYCLIST=1, CAR=2`), so the IVI and
the visualiser need no changes.
