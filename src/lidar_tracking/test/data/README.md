# Golden vectors

Generated from the reference Python implementation in `reference/` by
`../gen_goldens.py`. Regenerate with:

    python3 src/lidar_tracking/test/gen_goldens.py .

These are the oracle for the C++ port. They are deterministic (seeded RNG), so a
diff in regenerated output means the reference changed.

| file | pins |
|---|---|
| `corners.json` | `bbox3d2corners()` — the IVI-facing pose code |
| `metrics.json` | `giou_3d`, `iou_3d`, `dist_3d` over 24 box pairs (9 overlapping) |
| `orientation.json` | `within_range` and `orientation_correction` branch boundaries |
| `kalman.json` | KF init defaults + predict/update sequence (state and P diagonal) |
| `sequences.json` | four full `Track3D` scenarios: ID assignment and lifetime |

## Two conventions these lock down

Both verified against `corners.json` using the axis-aligned box
`[x=0, y=0, z=0, w=4.0, l=2.0, h=1.5, yaw=0]`:

- **`dims[0]` (`w`) is the along-heading (x) extent.** Its corners span
  x in [-2, 2], y in [-1, 1]. The detector's `Bndbox.w` is also the
  along-heading length (3.9 for the KITTI Car anchor), so dimensions copy
  straight across with no swap.
- **`z` is the box BOTTOM,** not its centre: corners span z in [0, 1.5] for
  h = 1.5. The detector decodes a centre z, so the adapter subtracts h/2.

`kalman.json` pins `P0_diag = [10]*7 + [10000]*3`, which comes from
`P[7:,7:] *= 1000` and `P *= 10` in `kalman_filter.py`. (Those two scalar
multiplies commute, so it is the resulting values that matter, not the order
the statements appear in.)

## What `sequences.json` is worth reading for

- `straight_line_car` — stable ID across 10 frames.
- `crossing_pedestrians` — two tracks pass through the same point at frame 6
  (y goes +3/-3 to -2.5/+2.5); IDs must NOT swap.
- `yaw_flip` — the detected heading alternates by ~pi, as the PointPillars
  direction classifier does in practice. This is the ONLY scenario that
  exercises `orientation_correction` inside the KF update; without it the other
  three all still pass.
- `dropout_and_return` — detections vanish for frames 5-7. Frame 5 still emits a
  box at x=14.0: that is the constant-velocity Kalman prediction. The track then
  dies (`max_age=2`) and the object re-births with a NEW id at frame 10 after
  `min_hits=3` is satisfied again.
