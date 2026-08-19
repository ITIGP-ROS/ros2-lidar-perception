# Status: validating the C++/TensorRT pipeline against the bag

Bag: `2011_09_29_drive_0004_sync_bag` (339 × `/kitti/velo`, `velo_link`,
`point_step=32`, intensity already in `[0, 0.99]` so `intensity_scale: 1.0`).

## Final result: end-to-end equivalence, not just tensor-level

The tensor-level agreement below was necessary but not sufficient — it showed
the network was right while the **decode** still had four real defects. Those
are found, fixed and measured; see `docs/COMPARISON.md` for the head-to-head.

| | value |
|---|---:|
| OLD published detections reproduced within 0.5 m | **94.0%** |
| median 3D centre error | **0.0001 m** |
| median z offset | **-0.0000 m** |
| class agreement | **99.8%** |

The four decode defects were: missing `nms_pre`/`max_num` caps, argmax-only
class emission, endpoint-to-endpoint anchor grid instead of bin centres, and a
centre-vs-bottom z origin. Each is described in `docs/COMPARISON.md`.

## Settled facts

| finding | evidence |
|---|---|
| Voxelisation is **correct** | pillar 0 point `y=-37.167` → `voxel_idy = floor((-37.167+39.68)/0.16) = 15` ✓, `x=0.171` → `voxel_idx = 1` ✓ |
| This scene needs **19 647 pillars** | measured with 40000 capacity |
| NVIDIA's reference ONNX accepts only **10 000** | `voxels[10000,32,10]`, sha256 `b48eada7…` |
| ⇒ NVIDIA's model **cannot** work here | it truncates ~half the scene, and truncation follows grid order (low `voxel_idy` = most negative y) — which is exactly the observed all-negative-y symptom |
| `MAX_VOXELS = 40000` was **my error** | it came from the PyTorch pipeline's `max_voxels=(16000, 40000)`, unrelated to the ONNX. The colleague's export was fine. |
| The model responds **weakly** on this data | PyTorch reference: max sigmoid 0.49, nothing >0.5, Pedestrian 0.49 vs Car 0.30. So "everything is a Pedestrian" is the model's real behaviour, not a decode bug. |
| `score_thresh: 0.1` was admitting noise | 464 cells >0.1 vs 13 cells >0.3. Raising to **0.35** gives 5–13 objects/frame with scores 0.35–0.67. |

## The tracker works

At `score_thresh 0.35`, over 40 frames: 87 track ids, **74 living ≥4 frames**,
longest 9. Track 114 moves 12.49 m in a straight line at constant `y ≈ -13.0`:

```
frame 14 ( 9.73,-13.04)  15 ( 8.24,-13.02)  16 ( 6.75,-13.00)  17 ( 5.25,-12.98)
frame 18 ( 3.54,-13.06)  19 ( 1.95,-12.97)  20 ( 0.38,-12.96)
```

That is physically coherent motion under one stable id through the whole C++
chain. **But it only proves the tracker is self-consistent on whatever it is
fed — NOT that the detector matches PyTorch.**

## RESOLVED — the detector DOES match PyTorch

Compared the **raw network output tensors** for a real bag cloud (19 663
pillars): TRT pipeline vs the PyTorch reference, same cloud, same weights.

| tensor | correlation | mean abs diff | range |
|---|---|---|---|
| `cls_preds` | **0.999911** | 0.0199 | -85.06 … -0.04 |
| `box_preds` | **0.999647** | 0.0024 | -1.38 … 1.71 |
| `dir_cls_preds` | **0.998964** | 0.0219 | -10.30 … 10.27 |

`argmax` of `cls_preds` lands on the **same cell (361674)** in both. `box_preds`
agrees to ~0.08% of its range. These are FP32 kernel-ordering differences, not
behavioural ones.

**Conclusion: the CUDA preprocessing (voxelisation + the 9 mmdet3d pillar
features) and the TensorRT engine faithfully reproduce the PyTorch model on real
bag data.**

Reproduce with:
```bash
PP_DUMP_CLS=<dir> ./pc_process --ros-args -p model_path:=.../pointpillar_gp.onnx \
    -p input_topic:=/kitti/velo -p intensity_scale:=1.0 -p class_remap:="[0,1,2]"
```
then diff `cls.bin`/`box.bin`/`dir.bin` against the PyTorch forward pass.

## Superseded: the earlier 6% detection-match figure

An earlier comparison matched only 6% of *decoded detections* within 1 m. That
number compared the shipped pipeline against **a Python decode + axis-aligned
NMS mirror written for the test** -- an approximation of
`postprocess_kernels.cu`, not the real thing, so the figure itself was not
measuring what it claimed to.

But the conclusion drawn at the time -- "attributable to the test-side mirror,
not to the pipeline" -- let the pipeline off too easily. The decode really did
disagree with the reference, in four separate ways, and it took a proper
head-to-head against the actual Python node to find them. Tensor-level
agreement says nothing about the decode that runs after the tensors.

## (historical) The earlier concern — detector does NOT match PyTorch

Compared TRT output against a PyTorch reference (numpy voxelisation + the 9
mmdet3d features + torch network + a decode mirroring `postprocess_kernels.cu`),
lock-step frame-aligned, 261 frames:

```
detections  TRT / PyTorch : 2417 / 1719      per-frame: 9.26 / 6.59
per-frame count correlation: 0.200
matched within 1.0 m       : 146/2417  (6.0%)
  median centre error      : 0.427 m
  median |score difference|: 0.0209
  class label agreement    : 95.9%
```

**6% is not equivalence.** A frame-offset sweep from -6 to +6 was FLAT (4.1–6.8%,
no peak), which rules out misalignment — the two genuinely disagree on where
objects are. Note the matched 6% agree well (0.43 m, score within 0.02, 96% same
class), so the two are not unrelated — they overlap partially.

Some excess in the TRT column is legitimate: its output is POST-tracking, so
AB3DMOT coasts tracks through misses and emits Kalman-smoothed positions rather
than raw detections. That cannot explain 94% unmatched.

## Reproducing the tensor comparison (this is the check that settled it)

Compare the **raw network tensors**, which bisects "preprocessing/inference
differs" from "decode differs":

```bash
PP_DUMP_CLS=/some/dir  ros2 run cuda_pointpillars_ros pc_process ...
# writes cls.bin / box.bin / dir.bin for the first frame
```

Then run the same cloud through the PyTorch reference and diff `cls_preds`.
- tensors match → the fault is in the CUDA **decode**
- tensors differ → the fault is in **preprocessing or the engine**

`PP_DEBUG_PILLARS=1` similarly dumps pillar counts and features.

## Performance (measured, not inferred)

```
generateVoxels 0.35 ms | generateFeatures 0.27 ms | TensorRT doinfer 20.25 ms
repack 124k pts 0.4 ms | full callback 22.0 ms  ->  ~45 Hz
```

GPU-bound on inference, ~4.5x faster than the bag's 10 Hz. An earlier note in
this file claimed ~4 Hz; that was **wrong** -- it measured the Python MCAP
replay harness (~230 ms/frame to decompress and deserialise a 4 MB cloud), not
the pipeline. TensorRT is doing its job.

## Config used for the runs above

`config/gp_pytorch_kitti.yaml` — the repo's own `epoch_160.pth` exported by
`tools/export_pytorch_to_onnx.py`: 3 classes in message order
(Pedestrian, Cyclist, Car → `class_remap: [0,1,2]`), `PILLAR_FEATURE_LAYOUT:
mmdet3d9`, 40000 voxels, `SCORE_THRESH: 0.35`.
