# Training handoff — Livox Mid-360S PointPillars

**For whoever owns training in `Abdulrahman2200925/perception`.**

Good news first: your existing `configs/mid360_*.yaml` is **already almost right**.
Only **three** things need to change. Everything else is correct — leave it alone.

> **Read this first.** There is an open bug on the *deployment* side: the CUDA
> runtime currently produces incorrect detections even with NVIDIA's own reference
> KITTI model correctly matched to its config. It is being investigated separately
> and is **not** caused by anything in this document. Until it is fixed, do not
> treat "the deployed model detects badly" as evidence about your training.

---

## Why this document exists

The runtime decoder (`params.h` in the deployment repo) mirrors the config the
model was trained and exported with — class order, anchor geometry, voxel grid,
pillar capacity. When they disagree, **nothing crashes**: the detector emits
confident, plausible-looking boxes that are noise.

So the numbers below aren't preferences, they're a contract. The node now
cross-checks the loaded engine against `params.h` at startup and refuses to run
on a mismatch.

---

## The three changes

### 1. `configs/mid360_dataset.yaml` — narrow the range to forward-only

```yaml
# from
POINT_CLOUD_RANGE: [-40.96, -40.96, -0.5, 40.96, 40.96, 3.0]
# to
POINT_CLOUD_RANGE: [0, -20.48, -0.5, 40.96, 20.48, 3.0]
```

No detections are needed behind the vehicle, and the Mid-360S's useful range is
~40 m. This is the **biggest inference-latency lever** on the Orin:

| | grid | head | cells |
|---|---|---|---|
| current (360°) | 512×512 | 256×256 | 262 144 |
| **new (forward)** | **256×256** | **128×128** | **65 536** — 4× fewer |

Leave `VOXEL_SIZE`, `MAX_POINTS_PER_VOXEL` and `MAX_NUMBER_OF_VOXELS` as they are.

### 2. `configs/mid360_pointpillar.yaml` — two classes, in this order

```yaml
# from
CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']
# to
CLASS_NAMES: ['Pedestrian', 'Car']
```

…and delete the `Cyclist` block from `ANCHOR_GENERATOR_CONFIG`. **Order matters** —
it sets the output channel order, which the runtime maps onto the message enum.
Keep the `Pedestrian` and `Car` anchor blocks exactly as they are; their sizes and
bottom heights are already correct for the 0.30 m mount.

### 3. Export the ONNX with `max_voxels` matching `MAX_NUMBER_OF_VOXELS.test`

```
inputs   voxels     [N, 32, 10]      N == MAX_NUMBER_OF_VOXELS.test
         voxel_idxs [N, 4]
         voxel_num  [1]
outputs  cls_preds      [1, 128, 128,  8]   # num_anchors(4) * num_classes(2)
         box_preds      [1, 128, 128, 28]   # num_anchors(4) * 7
         dir_cls_preds  [1, 128, 128,  8]   # num_anchors(4) * 2
```

The value of `N` is free — NVIDIA's own KITTI reference model ships `10000`. What
matters is that the export and the config **agree**. (An earlier draft of this
document claimed 40000 was required; that was wrong. 40000 came from the old
PyTorch pipeline's `max_voxels=(16000, 40000)`, which is unrelated to the ONNX.)

---

## Do NOT change these

| setting | value | why it's load-bearing |
|---|---|---|
| `USE_ABSLOTE_XYZ` | `True` | gives the **10-channel** pillar layout the CUDA kernel consumes. `False` gives 7 and nothing works. |
| `VOXEL_SIZE` | `[0.16, 0.16, 3.5]` | z size must equal the z range span |
| `MAX_POINTS_PER_VOXEL` | `32` | fixed in the CUDA kernel |
| anchor sizes / bottom heights | as-is | already re-based for the 0.30 m mount |
| `anchor_rotations` | `[0, 1.57]` | the runtime hardcodes 2 rotations per class |

**Stay on OpenPCDet.** Its `PillarVFE` is what the CUDA runtime is built against.
The old `epoch_160.pth` uses a 9-feature VFE variant and cannot be deployed
without extra conversion work — that is why it is being retired, not extended.

---

## Three things we need from you

**1. How is intensity normalised during training?** Raw `[0, 255]`, divided by 255,
or something else?

The node has an `intensity_scale` parameter that must match. It looks like a
sensor setting but it is a *training* convention, and a mismatch degrades accuracy
silently rather than failing. KITTI ships intensity already in `[0, 1]` (measured
`[0.000, 0.990]`), so KITTI uses `1.0`; a raw Livox reflectivity channel needs
`255.0`.

**2. How does the anchor generator place the grid?** Two conventions exist and
the runtime decoder must be told which one you used, or every predicted box
lands with a systematic offset (~0.08 m on the KITTI grid — enough to matter,
small enough that nothing crashes):

| convention | formula | who uses it | our setting |
|---|---|---|---|
| bin **centres** | `min + (i + 0.5) * span / n` | this repo's PyTorch model (`Anchors.get_anchors`) | `ANCHOR_ALIGN_CENTER: true` |
| **endpoint** to endpoint | `min + i * span / (n - 1)` | OpenPCDet default (`align_center: False`), NVIDIA's shipped model | `ANCHOR_ALIGN_CENTER: false` |

`config/livox.yaml` is set to **`false`**, because this document commits the
Livox model to OpenPCDet and `align_center: False` is its default. **If you turn
`align_center` on, say so** — we flip one line. Note this is deliberately the
opposite of `config/gp_pytorch_kitti.yaml`, which describes the old
mmdet3d-lineage model.

**3. What are your inference-time candidate caps?** `config/livox.yaml` carries
`NMS_PRE: 4096` and `MAX_NUM: 500`, OpenPCDet's `NMS_PRE_MAXSIZE` /
`NMS_POST_MAXSIZE` defaults. If your inference config differs, tell us — getting
these wrong does not crash, it just changes how many boxes come out. On the KITTI
model, omitting them entirely more than doubled the box count.

---

## Verifying before you hand a model back

1. **Regenerate the decoder from the config** — never hand-edit `params.h`:
   ```bash
   python3 tools/gen_params_h.py config/livox.yaml \
       -o src/cuda_pointpillars_ros/include/params.h
   ```
2. **Let the node check the engine.** It validates every tensor shape and the
   class count at startup and refuses to run on a mismatch, naming what differs:
   ```
   [FATAL] - voxels: engine [10000,32,10] but params.h implies [40000,32,10]
   ```
   A model that starts the node cleanly has the right geometry.
3. **Sanity-check classes.** The network should emit `0 = Pedestrian`, `1 = Car`;
   the node remaps those to the message's `PEDESTRIAN=0` / `CAR=2`.

---

## Reference: the full target config

```yaml
CLASS_NAMES: ['Pedestrian', 'Car']

POINT_CLOUD_RANGE: [0, -20.48, -0.5, 40.96, 20.48, 3.0]
VOXEL_SIZE:        [0.16, 0.16, 3.5]
MAX_POINTS_PER_VOXEL: 32
MAX_NUMBER_OF_VOXELS: {train: 16000, test: 40000}   # any N; must match the export

USE_ABSLOTE_XYZ: True
USE_NORM: True
NUM_FILTERS: [64]
used_feature_list: ['x', 'y', 'z', 'intensity']

ANCHOR_GENERATOR_CONFIG:
  - class_name: 'Pedestrian'
    anchor_sizes:          [[0.8, 0.6, 1.73]]
    anchor_rotations:      [0, 1.57]
    anchor_bottom_heights: [-0.10]
    matched_threshold: 0.5
    unmatched_threshold: 0.35
  - class_name: 'Car'
    anchor_sizes:          [[3.9, 1.6, 1.56]]
    anchor_rotations:      [0, 1.57]
    anchor_bottom_heights: [-0.19]
    matched_threshold: 0.6
    unmatched_threshold: 0.45
```
