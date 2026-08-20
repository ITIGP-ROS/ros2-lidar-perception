# Jetson Orin NX benchmark

Everything here is measured, not estimated, on the actual target — the prior
`docs/OPTIMIZATIONS.md` and `docs/COMPARISON.md` numbers are workstation-only
(RTX 3050 laptop, FP32). This is the first on-target run.

- **Board** Jetson Orin NX, 6 CPUs, GPU compute capability 8.7, 7608 MB
- **Precision** FP16 (`__aarch64__`-gated path)
- **TensorRT** 10.3.0.30, engine keyed on model hash + TRT version + `sm87` + precision
- **Bag** `2011_09_29_drive_0004_sync_bag`, 339 × `/kitti/velo`, 113 622–124 910
  pts/cloud (mean 119 239)
- **Method** real ROS node (`cuda_pointpillars_ros`/`pc_process`), `PP_TIME_CB=1`,
  wall clock inside the subscriber callback — same method as `docs/OPTIMIZATIONS.md`
- **Build** Yocto/BitBake recipe build, `CMAKE_BUILD_TYPE=Release`, default
  (`PERFORMANCE_LOG` off — see §6)
- **OS** custom Yocto image, BusyBox userspace — no `tegrastats`, no `nvidia-smi`

---

## 1. Headline

| | Workstation (RTX 3050, FP32) | Jetson Orin NX (FP16) |
|---|---:|---:|
| callback, median | **20.80 ms** | **31.20 ms** |
| throughput (from median) | **48.1 Hz** | **32.1 Hz** |
| sustained, live, at 30 Hz demand | — | **28.87 Hz** |
| headroom over the 10 Hz sensor rate | 4.8× | **3.2×** |

FP16 on Orin does not close the gap to the discrete laptop GPU — Orin NX's raw
compute is well below an RTX 3050 even with the precision advantage. It does not
need to: 3.2× headroom over the sensor's actual 10 Hz is comfortable, and the
30 Hz-demand test below confirms this is a real, sustained number, not a median
that hides backlog.

## 2. Per-frame timing distribution

Single real-time pass (`--rate 1.0`), n=291 frames.

| stage | min | median | mean | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| repack (CPU) | 0.30 | 0.30 | 0.32 | 0.40 | 0.50 | 0.70 |
| infer + track + publish | 28.20 | 30.80 | 30.73 | 34.40 | 35.90 | 37.80 |
| **total** | **28.50** | **31.20** | **31.06** | **34.70** | **36.20** | **38.10** |

All ms. Repack is noise (<1% of the frame, matching the workstation figure of
0.3 ms exactly). The frame is dominated by `infer+track+publish` — consistent
with the workstation breakdown where `doinfer` alone was 92% of the callback.
That per-stage GPU split was not re-measured here (see §6).

## 3. Sustained throughput at 3× demand

`ros2 bag play --rate 3.0` (30 Hz demand) against the same node, n=307 frames.

| | value |
|---|---:|
| achieved rate | **28.87 Hz** |
| median total per callback | 31.2 ms (unchanged from the 1× run) |
| median inter-message gap | 33.9 ms |

The per-callback cost is identical to the 1× run — this is a genuinely
GPU-bound, saturating workload, not a number that degrades under backlog. The
achieved rate (28.87 Hz) sits just under the 30 Hz demand because the median
inter-message gap (33.9 ms) is set by processing time, not by the requested
rate; this matches the §1 throughput estimate (32.1 Hz from the median) within
measurement noise.

## 4. TensorRT engine build / cache

| | time |
|---|---:|
| cold build (cache deleted, from `.onnx`) | **215 s (3.6 min)** |
| cached load (engine present) | **< 1 s** |
| engine file | 9.93 MB, fp16, sm87, TRT 10.3 |

Cache path: `<model>.onnx.<hash>.trt10.3.sm87.fp16.engine`, next to the `.onnx`.
Confirms the engine only needs building once per (model, TRT version, GPU arch,
precision) tuple — see `docs/OPTIMIZATIONS.md` §3.1 for why this matters.

## 5. Resource footprint (steady state)

| | value |
|---|---:|
| `pc_process` RSS | **1.76 GB** (1 846 004 kB) — includes CUDA/TensorRT allocations; Orin's unified memory puts these in process RSS, unlike a discrete-GPU box |
| host CPU | **~4% of one core** (6 available) — orchestration only; inference is GPU-offloaded |
| GPU clock under load | pinned at max, **1.173 GHz** — no throttling observed over the test window |
| GPU utilization % | **not reliably obtained** — see below |

`tegrastats` and `nvidia-smi` are both absent from this minimal Yocto image.
`/sys/class/devfreq/17000000.gpu/device/load` exists and is readable, but
returned inconsistent values across repeated samples during identical sustained
load (37, 45, 496, 45, 37, 311) — out of a sane 0–100 range on more than one
sample, so it is not trusted here rather than reported as fact. GPU clock
(`cur_freq` == `max_freq` throughout) is the one GPU-side number treated as
solid.

## 6. What this benchmark does not cover

- **Per-stage GPU breakdown** (`generateVoxels` / `generateFeatures` / `doinfer`
  split). Requires a `-DPERFORMANCE_LOG=1` rebuild, not done this pass. The
  workstation figure (`doinfer` = 92% of the frame) is likely representative but
  unverified on Orin.
- **Real GPU utilization %** — blocked by the missing `tegrastats`/`nvidia-smi`
  and the untrustworthy devfreq `load` node (§5).
- **Pillar count** for this bag on Orin — not captured (`PP_DEBUG_PILLARS` or
  `PERFORMANCE_LOG` not enabled).
- **Livox model / bag.** This run used the KITTI-trained model against KITTI
  data (`profile:=kitti`). The Livox path (`config/livox.yaml`,
  `docs/TRAINING_HANDOFF.md`) is untested on-target.
- **INT8** — not attempted.

## 7. A DDS discovery note, not a processing issue

Starting `ros2 bag play` as a fresh process against an already-running
subscriber cost the first **~48 of 339** `/kitti/velo` messages (~4.8 s) before
the two DDS participants finished matching — visible as a shorter effective
capture window (291 messages instead of 339) with **zero gaps > 150 ms**
anywhere in the remaining stream once matched, sustaining 9.76 Hz against the
bag's ~9.6–10 Hz native rate. This is discovery latency on a brand-new
publisher, not the pipeline falling behind — median total processing time
(31 ms) leaves 69 ms of slack in every 100 ms sensor period. Worth knowing for
any cold-start-sensitive scenario (a bag replay, or a sensor driver restarting
mid-mission); not a pipeline defect.

## 8. Reproduce

```bash
# on the Jetson, terminal 1
. /opt/ros/humble/setup.bash
PP_TIME_CB=1 ros2 launch lidar_perception_bringup perception.launch.py \
    profile:=kitti visualize:=false \
    model_path:=/data/lidar-perception/model/pointpillar_gp.onnx

# terminal 2, once "model ready" appears in terminal 1
ros2 bag play /root/bags/2011_09_29_drive_0004_sync_bag --rate 1.0   # or --rate 3.0

# aggregate: grep terminal 1's output for "CB: repack" and compute
# min/median/mean/p95/p99/max on the repack/infer+track+publish/total fields
```

Cold-build timing: delete the cached `*.engine` file next to the `.onnx` before
launching, and time from process start to the `model ready` log line.
