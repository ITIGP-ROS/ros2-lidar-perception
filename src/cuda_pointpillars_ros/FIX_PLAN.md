# FIX_PLAN.md — CUDA-PointPillars-ROS2 on Jetson Orin NX (aarch64)

Phase 1 investigation only. No files in this repo were modified. All paths below
were verified against the running container `compassionate_bassi`
(image `dustynv/ros:humble-llm-r35.4.1`, container ID `f3bb913a71ed`) on host
`es-iti@10.42.0.115`, not assumed.

---

## 1. CMakeLists.txt — x86_64 → aarch64 path problems

### 1.1 CUDA target dir (line 17)

```cmake
set(CUDA_INSTALL_TARGET_DIR targets/x86_64-linux)
```

Verified: `ls /usr/local/cuda/targets/` → only `aarch64-linux` exists. `x86_64-linux`
does not exist, so `CUDA_INCLUDE_DIRS` (line 19, built from this var) currently
points at a nonexistent directory.

**Fix:**
```cmake
set(CUDA_INSTALL_TARGET_DIR targets/aarch64-linux)
```

### 1.2 TensorRT include dir (line 41)

```cmake
set(TENSORRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu/)
```

Verified: `find / -name NvInfer.h` → `/usr/include/aarch64-linux-gnu/NvInfer.h`.
TensorRT 8.5.2.2 headers live under the aarch64 multiarch include path, not
the x86_64 one (which doesn't exist in this container).

**Fix:**
```cmake
set(TENSORRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu/)
```

### 1.3 TensorRT library dir (line 42)

```cmake
set(TENSORRT_LIBRARY_DIRS /usr/lib/x86_64-linux-gnu/)
```

Verified: `find / -name "libnvinfer.so*"` →
`/usr/lib/aarch64-linux-gnu/libnvinfer.so{,.8,.8.5.2}` (same dir has
`libnvonnxparser.so{,.8,.8.5.2}`, needed by `target_link_libraries` at line 99).

**Fix:**
```cmake
set(TENSORRT_LIBRARY_DIRS /usr/lib/aarch64-linux-gnu/)
```

### 1.4 Leftover x86-host cross-compile sysroot flags (lines 28–37)

```cmake
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -ccbin ${CMAKE_CXX_COMPILER}
    -Xcompiler -DWIN_INTERFACE_CUSTOM
    -Xcompiler -I/usr/aarch64-linux-gnu/include/
    -Xlinker -lsocket
    -Xlinker -rpath=/usr/lib/aarch64-linux-gnu/
    -Xlinker -rpath=/usr/aarch64-linux-gnu/lib/
    -Xlinker -L/usr/lib/aarch64-linux-gnu/
    -Xlinker -L/usr/aarch64-linux-gnu/lib/
)
```

This block is clearly written for cross-compiling **from an x86_64 host** to
aarch64 (NVIDIA's original CUDA-PointPillars supports this workflow), where
`/usr/aarch64-linux-gnu/` is the cross sysroot. We are building **natively on
the Jetson**, and:

- Verified: `/usr/aarch64-linux-gnu/` does **not exist** in this container
  (`ls` → "No such file or directory"). So `-I/usr/aarch64-linux-gnu/include/`,
  `-rpath=/usr/aarch64-linux-gnu/lib/`, and `-L/usr/aarch64-linux-gnu/lib/` are
  all dead paths — harmless (nvcc/gcc silently ignore nonexistent `-I`/`-L`/`-rpath`
  entries) but should be removed as cleanup.
- Verified: `/usr/lib/aarch64-linux-gnu/` (the **native** multiarch path, no
  separate `aarch64-linux-gnu` top-level dir) does exist and is correct to keep.
- `-Xlinker -lsocket`: there is no `libsocket.so`/`.a` anywhere in this
  container (`find / -iname "libsocket*"` only turns up an unrelated Samba
  plugin, `libsocket-blocking.so.0`). On Linux, socket functions live in libc;
  `-lsocket` is a Solaris-ism. This **may cause `ld: cannot find -lsocket`** at
  link time. Cannot confirm without actually linking (deferred to Phase 2
  build step), but flagged as the most likely build-breaking line in the
  whole file.

**Fix (proposed):**
```cmake
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -ccbin ${CMAKE_CXX_COMPILER}
    -Xcompiler -DWIN_INTERFACE_CUSTOM
    -Xlinker -rpath=/usr/lib/aarch64-linux-gnu/
    -Xlinker -L/usr/lib/aarch64-linux-gnu/
)
```
Risk: low for the sysroot lines (dead paths → no-ops either way), but verify
the build actually links once `-lsocket` is removed — keep removal isolated
to its own commit/step in Phase 2 so it's easy to bisect if something else
was secretly depending on it (unlikely, but cheap to be careful).

### 1.5 GENCODE / SM architectures (lines 21–26)

```cmake
set( SMS 30 32 35 37 50 52 53 60 61 62 70 72 75 87)
...
set(HIGHEST_SM 87)
```

Orin NX is `sm_87` (Ampere). `HIGHEST_SM 87` is already correct — the PTX
fallback (`compute_87`/`code=compute_87`) targets the right architecture.
Building cubins for SMs 30–75 is pure wasted compile time on this device
(those binaries will never be loaded at runtime here); it is not incorrect,
just slow.

**Recommendation:** trim to just the target device for faster iteration
during development:
```cmake
set( SMS 87)
```
Risk: none functionally (only affects build time, not correctness on this
device). If this package or `.engine`/`.cache` artifacts ever need to run on
a different GPU, the SM list would need that GPU's number added back —
worth a one-line comment if trimmed.

Note: the (uncommitted, remote-only) `origin/vis` branch added `89` to this
list — that's Ada Lovelace (desktop RTX 40-series / non-Orin), not applicable
here. Don't carry that over.

---

## 2. `autoware_auto_perception_msgs` dependency

### Verdict: **missing, not installable from the configured apt repo, and — critically — not actually used by any code that compiles into the `pc_process` binary on the current `main` branch.** Recommend: remove it.

**Presence check:**
```
$ ros2 pkg prefix autoware_auto_perception_msgs
Package not found
$ ros2 pkg list | grep -i autoware
(no output)
```

**Apt availability:**
```
$ cat /etc/apt/sources.list.d/*.list
deb [arch=arm64 signed-by=...] http://packages.ros.org/ros2/ubuntu focal main
```
Only the standard ROS 2 apt repo is configured — no Autoware Foundation repo.
`apt-cache search autoware` and `apt-cache search ros-humble-autoware` both
returned nothing, but the local apt cache also appears unpopulated
(`apt-cache search ros-humble` → 0 results), so that result alone isn't
conclusive — `apt-get update` would be needed for a definitive apt answer,
and Phase 1 is read-only/no-network-mutation, so this wasn't run. That said,
`autoware_auto_perception_msgs` is part of the (now-deprecated) Autoware.Auto
message set, which has never been published through `packages.ros.org`'s
default index for any distro — it's normally obtained by building
`autoware_msgs`/`autoware_auto_msgs` from source inside a full Autoware
workspace. Treat "installable via apt" as **unlikely** even after
`apt-get update`.

**Actual usage in the code that builds today (`main` branch, current working
tree) — verified by reading every reference:**

- `src/pc_process.cpp`: the only references are commented out:
  ```cpp
  // #include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>
  ...
  // rclcpp::Publisher<autoware_auto_perception_msgs::msg::PredictedObjects>::SharedPtr pub_;
  ...
  // pub_ = this->create_publisher<autoware_auto_perception_msgs::msg::PredictedObjects>("objects", 10);
  ```
- `src/cuda_pp_ros.cpp` / `include/cuda_pointpillars_ros/cuda_pp_ros.hpp`: no
  reference at all.
- **The node currently publishes nothing.** `infer()` runs inference and only
  prints `"Bndbox objs: N"` to stdout; there is no active publisher.

So on `main`, `autoware_auto_perception_msgs` is referenced **only** in
`CMakeLists.txt` (`find_package(... REQUIRED)` line 12, and
`ament_target_dependencies(...)` line 90) — a pure vestige. Removing it from
CMakeLists (and from `package.xml`, see §5) requires **zero** C++ changes to
get back to "compiles, runs, prints box count."

**A relevant discovery — `origin/vis` remote branch:** this repo has a remote
branch `origin/vis` (not checked out; `main` is currently checked out) whose
last commit message is *"Build and test complete, next: refine Algorithm"*.
It already replaced the dead autoware publisher with a working
`visualization_msgs::msg::MarkerArray` publisher:

```cpp
// src/pc_process.cpp on origin/vis
rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_;
...
pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("objects_marker", 10);
```
```cpp
// src/cuda_pp_ros.cpp on origin/vis — infer() now returns a MarkerArray
visualization_msgs::msg::MarkerArray infer(...) {
  ...
  for (auto &pred : nms_pred) {
    obj.header.frame_id = "/rslidar";
    obj.type = visualization_msgs::msg::Marker::CUBE;
    obj.pose.position.x/y/z = pred.x/y/z;
    obj.scale.x/y/z = pred.l/w/h;
    ... // color, alpha, id
    objs.markers.push_back(obj);
  }
  return objs;
}
```
Note: `origin/vis` still nominally lists `autoware_auto_perception_msgs` in
`find_package`/`package.xml`/an unused `#include` — but it is dead weight
there too (the actual publisher type is `visualization_msgs::msg::MarkerArray`,
not any autoware type). This independently confirms that dropping autoware
entirely and standing up `visualization_msgs::msg::MarkerArray` output is a
proven, low-risk path — someone already built and tested it.

**Recommendation for Phase 2:** remove `autoware_auto_perception_msgs`
everywhere (CMakeLists, package.xml) and adopt the `origin/vis` approach of
publishing `visualization_msgs::msg::MarkerArray` (ships with every ROS 2
install, viewable directly in RViz2, no extra dependency). This is strictly
better than today's main (which publishes nothing) and removes the one
dependency that cannot currently be satisfied.

---

## 3. Point-field parsing in the inference path (`src/cuda_pp_ros.cpp`)

**Where it happens:** `void infer(sensor_msgs::msg::PointCloud2::SharedPtr msg)`
in `src/cuda_pp_ros.cpp`:

```cpp
float* points = (float*)msg->data.data();
size_t height = msg->height;
size_t width = msg->width;
size_t row_step = msg->row_step;
size_t length = row_step * height;
size_t points_size = length/sizeof(float)/4;
...
checkCudaErrors(cudaMemcpy(points_data, points, points_data_size, cudaMemcpyDefault));
...
pointpillar.doinfer(points_data, points_size, nms_pred);
```

And in `src/preprocess_kernels.cu`, the CUDA kernel that consumes this buffer:
```cpp
float4 point = ((float4*)points)[point_idx];
...
atomicExch(address+3, point.w);   // 4th value stored straight through, no scaling
```

**Finding: there is no PointField-aware parsing anywhere in the inference
path.** `msg->fields` (the array of `{name, offset, datatype}` describing
what's actually in the byte buffer) is **never read**. The code just
reinterprets the raw `msg->data` bytes as a flat array of `float4`s — i.e. it
hard-assumes:
- exactly 4 fields per point, all `float32`,
- tightly packed with no padding (`point_step == 16` bytes),
- field order is `(x, y, z, intensity)`,
- intensity already normalized to roughly KITTI's 0–1 range (no scaling is
  ever applied to `point.w`).

(The only other place in the repo that touches "fields" by name is
`src/pillarScatter.cpp:273`, `const char* attr_name = fields[i].name;` — that
is a TensorRT plugin's `PluginFieldCollection` for the custom PillarScatter
layer, completely unrelated to `sensor_msgs::PointField`. There's also
`src/pc_info.cpp`, a debug-only executable that's commented out of the build
in CMakeLists — it uses the deprecated
`sensor_msgs::convertPointCloud2ToPointCloud()`, which only extracts x/y/z and
silently drops intensity entirely, so it's not a usable starting point
either.)

**Why this matters for the real rosbag:** the bag's cloud has fields
`x, y, z, reflectivity, tag` — different field count, different name for the
4th channel, and (per the task brief) `reflectivity` in 0–255 rather than a
normalized 0–1 range. Depending on the driver's actual `PointField` layout
(typical RoboSense-style drivers — and the hardcoded subscription topic name
is literally `"rslidar_points"`, see below — publish `tag` as `uint8` and
`reflectivity` as `float32`, which makes `point_step` something other than
16 bytes, e.g. 17 or padded to a multiple of 4/8). Feeding that buffer through
the current `(float4*)points` cast will not crash — it will silently
mis-align every point's x/y/z/"w" by walking the wrong stride, and even if it
happened to align it would feed raw 0–255 reflectivity straight into the
network where it expects ~0–1 intensity. This is the highest-risk, easy-to-miss
bug in the whole port: it fails by producing **wrong detections**, not by
crashing or erroring.

```cpp
// src/pc_process.cpp — subscription topic is hardcoded already to match
// a RoboSense-style driver:
sub_ = this->create_subscription<PointCloud2>("rslidar_points", 10, ...);
```

**Required Phase 2 fix (highest-effort item):** stop treating `msg->data` as
a flat `float4` array. Use `msg->fields` to locate the actual byte offsets
for `x`, `y`, `z`, `reflectivity` (e.g. via
`sensor_msgs::PointCloud2Iterator<float>` per-field iterators, or by manually
scanning `msg->fields[i].name`/`.offset`/`.datatype` once and caching the
offsets), then build a **new**, tightly-packed `float32 x4` buffer
`(x, y, z, reflectivity/255.0f)` per point — matching what the model
actually expects — before handing it to `pointpillar.doinfer()`. This also
needs to tolerate `tag` (and whatever else) being present in between offsets
that we should skip over, and should not assume `point_step == 16` anymore.

---

## 4. Hardcoded model path (bonus finding, not in the original ask but blocks startup)

`src/cuda_pp_ros.cpp`:
```cpp
std::string Model_File = "/home/e404/perception_ws/src/cuda_pointpillars_ros/model/pointpillar.onnx";
```
Verified: this path does **not exist** in the container (`ls` →
"No such file or directory"). It's the original author's x86 dev-machine home
directory. The real model is at
`/workspace/downloads/CUDA-PointPillars-ROS2/model/pointpillar.onnx` (19 MB,
confirmed present).

This is not cosmetic — it's a hard startup crash. In `src/pointpillar.cpp`,
`TRT::TRT(...)`, if no `.cache` file is found next to `modelFile` it tries to
parse the onnx at that path, and:
```cpp
if (!parser->parseFromFile(modelFile.data(), ...))
{
    std::cerr << ": failed to parse onnx model file, ...";
    exit(-1);
}
```
i.e. the whole node process calls `exit(-1)` immediately on startup with the
current hardcoded path. **This must be fixed in Phase 2 or the node will
never run at all**, independent of every other fix in this document.

Minimal fix: update the hardcoded string to the real path. Better fix (more
robust, slightly more work): resolve it at runtime via
`ament_index_cpp::get_package_share_directory("cuda_pointpillars_ros")` +
`/model/pointpillar.onnx`, once `model/` is added to the `install()` rules in
CMakeLists (it currently is not — only the `pc_process` binary is installed).
Recommend the minimal fix first to unblock everything else, and treat the
share-dir approach as a stretch goal.

---

## 5. `package.xml`

**Finding: there is no `package.xml` anywhere in the working tree** —
confirmed via `find ... -iname package.xml` (no results) and
`git ls-files | grep -i package` on `HEAD` (no results). `ament_cmake`
packages require a `package.xml`; without one, `colcon build` will not
recognize this directory as a buildable package at all, regardless of any
CMakeLists fix.

This is **not** a "the author forgot it" situation — it's a branch-tracking
artifact. The currently checked-out branch is `main`. A different,
**not-checked-out** remote branch, `origin/vis` (ahead of `main` by 2
commits, last one *"Build and test complete, next: refine Algorithm"*), does
have a `package.xml`, added in commit `1a30bed` ("add visualization"):

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>cuda_pointpillars_ros</name>
  <version>0.0.0</version>
  <description>TODO: Package description</description>
  <maintainer email="wlyzhcn@qq.com">e404</maintainer>
  <license>TODO: License declaration</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <depend>rclcpp</depend>
  <depend>sensor_msgs</depend>
  <depend>autoware_auto_perception_msgs</depend>
  <depend>visualization_msgs</depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

**Recommendation:** adapt this for Phase 2 — same shape, but drop the
`<depend>autoware_auto_perception_msgs</depend>` line per §2's recommendation
(keep `visualization_msgs`, which is already listed here and which we're
standardizing on for output).

Also worth noting: `model/pointpillar.onnx` is currently tracked as a 133-byte
placeholder at `main`'s `HEAD` (`git diff` shows
`model/pointpillar.onnx | Bin 133 -> 19343782 bytes`) — i.e. the real 19 MB
ONNX file already on disk was dropped in manually on top of the `main`
checkout and was never committed. Not a problem for building (the file is
there and that's what matters), just noting it so Phase 2 doesn't get
confused by `git status` showing it as "modified."

---

## 6. Workspace / colcon layout

`/workspace/downloads/CUDA-PointPillars-ROS2` is laid out correctly **as a
single ROS 2 package** (CMakeLists.txt + `src/` + `include/` at the package
root) — that part is fine. What it is **not** is a colcon **workspace**: a
colcon workspace needs a root directory containing a `src/` folder whose
*subdirectories* are packages (each with their own `package.xml` +
`CMakeLists.txt`). This repo's own `src/` holds this package's `.cpp`/`.cu`
files directly — that's the package's source folder, not a workspace's
package-container folder. Running `colcon build` directly inside
`/workspace/downloads/CUDA-PointPillars-ROS2` would have colcon scan `src/`
for nested packages, find none, and build nothing.

Verified there is no pre-existing colcon workspace elsewhere in the container
that already expects this package (searched for `src/` directories matching
a workspace pattern — only unrelated ones exist, e.g. `/opt/piper/src`,
`/opt/jetson-inference/ros/src`). `/opt/ros/humble/` itself is install-space
only (`install/`, `log/`, no `src/`) — that's the base ROS distro, not a
place to add this package.

**For Phase 2:** create a workspace, e.g. `/workspace/perception_ws/src/`,
and place (symlink or move) this repo at
`/workspace/perception_ws/src/CUDA-PointPillars-ROS2`, then run
`colcon build` from `/workspace/perception_ws`. `colcon` itself is already
installed and on `PATH` (`/usr/bin/colcon`, plugins present and reasonably
up to date).

---

## 7. Toolchain sanity check (no issues found)

- `gcc`/`g++` → `gcc-9`/`g++-9` (Ubuntu 9.4.0), both at `/usr/bin/gcc` and
  `/usr/bin/g++` as CMakeLists already expects (lines 15–16) — fine as-is,
  no change needed, these paths are arch-independent.
- `nvcc` → CUDA 11.4.315, matches the task brief.
- `cmake` → 3.29.3, satisfies `cmake_minimum_required(VERSION 3.8)`.
- TensorRT → 8.5.2.2 (libnvinfer.so.8.5.2 / libnvonnxparser.so.8.5.2), well
  above the README's stated `>= 8.4` requirement.

---

## Ordered list of changes for Phase 2, flagged by risk

1. **[Low risk]** CMakeLists.txt: `targets/x86_64-linux` → `targets/aarch64-linux` (§1.1).
2. **[Low risk]** CMakeLists.txt: `TENSORRT_INCLUDE_DIRS`/`TENSORRT_LIBRARY_DIRS`
   `x86_64-linux-gnu` → `aarch64-linux-gnu` (§1.2, §1.3).
3. **[Low risk]** CMakeLists.txt: drop the dead `/usr/aarch64-linux-gnu/...`
   cross-sysroot `-I`/`-rpath`/`-L` flags (§1.4). Verify at build time whether
   `-Xlinker -lsocket` must also be dropped (likely yes — no such lib exists).
4. **[Low risk]** Create `package.xml` adapted from `origin/vis`'s version,
   without the `autoware_auto_perception_msgs` depend (§5).
5. **[Low risk]** CMakeLists.txt: remove
   `find_package(autoware_auto_perception_msgs REQUIRED)` and drop it from
   `ament_target_dependencies(...)`; add `find_package(visualization_msgs REQUIRED)`
   and add `visualization_msgs` to `ament_target_dependencies(...)` (§2).
6. **[Low risk]** Fix the hardcoded `Model_File` path in `src/cuda_pp_ros.cpp`
   to point at the real on-disk model location (§4) — **blocks all runtime
   testing until fixed**, independent of everything else.
7. **[Low risk, optional]** Trim `SMS` to just `87` for faster builds (§1.5).
8. **[Medium risk]** Adopt the `origin/vis` `visualization_msgs::msg::MarkerArray`
   publisher pattern in `src/pc_process.cpp` / `src/cuda_pp_ros.cpp` /
   `include/cuda_pointpillars_ros/cuda_pp_ros.hpp` so the node actually
   publishes detections instead of only printing a count (§2). Medium risk
   only because it's new code being merged in from a different branch by
   hand, not because the approach is unproven (it was already "build and test
   complete" on `origin/vis`).
9. **[Medium risk]** Set up a proper colcon workspace
   (`/workspace/perception_ws/src/CUDA-PointPillars-ROS2`) and confirm
   `colcon build` discovers and builds the package once `package.xml` exists (§6).
10. **[High risk — most invasive, most important]** Rewrite the point parsing
    in `infer()` (`src/cuda_pp_ros.cpp`) to read `msg->fields` by name instead
    of blindly casting `msg->data` to `float4*`: extract `x`, `y`, `z`,
    `reflectivity`, normalize `reflectivity/255.0f`, pack into a fresh
    contiguous `float32 x4` buffer, and feed *that* to `pointpillar.doinfer()`
    (§3). This is the change most likely to have subtle bugs (wrong offset
    math, endianness/datatype mismatches) and the one most worth testing
    against a recorded sample of the actual rosbag before trusting any
    detection output.

Steps 1–7 are necessary just to get a successful `colcon build` and a process
that starts without crashing. Step 8 is necessary to get any usable output.
Step 10 is necessary to get *correct* output against the real sensor data —
without it, the package will build and run but silently produce garbage
detections.

---

## PHASE_2A_RESULT

Scope executed: get a successful `colcon build` and a node that starts
without crashing. Point-cloud field parsing and CUDA inference code were
**not** touched, and the subscription topic name (`rslidar_points`) was
**not** changed, per Phase 2a constraints.

### Step 0 — ONNX backup

```
cp model/pointpillar.onnx /workspace/pointpillar.onnx.bak
md5sum: bde8184e7dd17ae14c850e38b57ccfd4 (19 MB)
```
Verified `bde8184e7dd17ae14c850e38b57ccfd4` matched throughout every
subsequent step (post-checkout, post-build, post-smoke-test). The file was
never disturbed.

### Step 1 — Branch switch

`git checkout -b vis origin/vis` initially **refused**:
```
error: Your local changes to the following files would be overwritten by checkout:
        CMakeLists.txt, LICENSE, NOTICE, README.md,
        include/cuda_pointpillars_ros/cuda_pp_ros.hpp, model/pointpillar.onnx,
        src/cuda_pp_ros.cpp, src/pc_process.cpp, src/pointpillar.cpp
```
Inspected this first: every one of those "local changes" except
`model/pointpillar.onnx` was a pure file-mode diff (100644→100755, zero
content delta — confirmed via `git diff --stat` showing `0` insertions/
deletions for all of them). For `model/pointpillar.onnx`, `git hash-object`
on the working-tree file matched `origin/vis`'s tracked blob
(`e46689c4e4...`) exactly — i.e. the real model already on disk was already
byte-identical to what `vis` wants. So `git stash` (no `-u`, leaving the
untracked `FIX_PLAN.md` alone) followed by `git checkout -b vis origin/vis`
was clean and lost nothing meaningful. The stash was left undropped in case
anything needs to be referenced later, but nothing from it is needed.

Confirmed after checkout: `package.xml` exists (750 bytes), `src/pc_process.cpp`
uses `visualization_msgs::msg::MarkerArray`, and
`model/pointpillar.onnx` was untouched by the checkout (still 19 MB,
`bde8184e...`) — no LFS-pointer regression occurred (there's no git-lfs
binary in the container; git printed a harmless one-time notice about the
missing `git-lfs` post-checkout hook, but no LFS pointer file was ever
involved since the blob itself is the real 19 MB model, not a pointer).

### Step 2 — CMakeLists.txt fixes

`origin/vis` had **not** fixed any of the x86_64 paths — it had actually
regressed them further (pointed `TENSORRT_INCLUDE_DIRS`/`_LIBRARY_DIRS` at
`~/TensorRT-8.4.3.1/...`, a different nonexistent dev path, and added `89`
to the SM list). Applied, in one edit:
- `CUDA_INSTALL_TARGET_DIR`: `targets/x86_64-linux` → `targets/aarch64-linux`
- `TENSORRT_INCLUDE_DIRS`: `~/TensorRT-8.4.3.1/include` → `/usr/include/aarch64-linux-gnu/`
- `TENSORRT_LIBRARY_DIRS`: `~/TensorRT-8.4.3.1/lib` → `/usr/lib/aarch64-linux-gnu/`
- Removed dead cross-sysroot flags (`-I/usr/aarch64-linux-gnu/include/`,
  `-rpath=/usr/aarch64-linux-gnu/lib/`, `-L/usr/aarch64-linux-gnu/lib/`) and
  `-Xlinker -lsocket`; kept the native `-rpath=/usr/lib/aarch64-linux-gnu/`
  and `-L/usr/lib/aarch64-linux-gnu/`.
- Trimmed `SMS` to `87` only (was `30 32 35 37 50 52 53 60 61 62 70 72 75 87 89`),
  `HIGHEST_SM` to `87` (was `89`), with a one-line comment explaining why.
- Bonus fix (found while editing, in-scope as a "missing path" issue): line
  `include_directories(${CUDA_INCLUDE_DIRS}visualization_msgs include)` had
  `visualization_msgs` accidentally concatenated onto `${CUDA_INCLUDE_DIRS}`
  with no separator, and was missing `${TENSORRT_INCLUDE_DIRS}` entirely.
  Fixed to list `${CUDA_INCLUDE_DIRS}`, `${TENSORRT_INCLUDE_DIRS}`, `include`
  as separate lines.

The suspected `-lsocket` link failure (flagged as "most likely build-breaking
line" in §1.4) was never actually tested in isolation since it was removed
before the first build attempt — can't confirm it would have failed, but
removing it cost nothing and the build succeeded without it.

### Step 3 — autoware removal

`origin/vis`'s `CMakeLists.txt` still had
`find_package(autoware_auto_perception_msgs REQUIRED message_generation)`
and listed it in `ament_target_dependencies`; `package.xml` still had
`<depend>autoware_auto_perception_msgs</depend>`. Removed both, and confirmed
`visualization_msgs` was already present in both files (it was — `vis` added
it correctly when the MarkerArray publisher was written).

One thing beyond the letter of "CMakeLists.txt and package.xml" that turned
out to be necessary: `include/cuda_pointpillars_ros/cuda_pp_ros.hpp` and
`src/pc_process.cpp` both had an **active** (not commented-out)
`#include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>` that is
never actually used anywhere (the publisher type is, and always was in this
branch, `visualization_msgs::msg::MarkerArray` — `PredictedObjects` is never
referenced beyond the include line). Once `find_package`/
`ament_target_dependencies` no longer pull in that package's include path,
this `#include` would fail to resolve and break the build. Removed both
dead include lines. This is dependency cleanup, not point-parsing or
inference logic, so treated it as in-scope for "missing find_package"-class
fixes.

### Step 4 — Hardcoded model path

`origin/vis` had already moved the path from the Phase 1 report's
`/home/e404/...` to a different, still-nonexistent dev path,
`/home/wly/perception_ws/src/cuda_pointpillars_ros/model/pointpillar.onnx`.
Changed to the real container path:
```
/workspace/downloads/CUDA-PointPillars-ROS2/model/pointpillar.onnx
```
(Single-line `sed` substitution in `src/cuda_pp_ros.cpp`; nothing else in
that file touched.)

### Step 5 — colcon workspace

No pre-existing workspace. Created:
```
mkdir -p /workspace/perception_ws/src
ln -sfn /workspace/downloads/CUDA-PointPillars-ROS2 /workspace/perception_ws/src/CUDA-PointPillars-ROS2
```
Symlink, not a copy/move — the original directory is still the single
source of truth, `git` operations there are unaffected by the symlink.

### Step 6 — Build result: **SUCCESS**

```
$ cd /workspace/perception_ws && source /opt/ros/humble/install/setup.bash
$ colcon build --packages-select cuda_pointpillars_ros
Starting >>> cuda_pointpillars_ros
Finished <<< cuda_pointpillars_ros [29.8s]
Summary: 1 package finished [30.0s]
```
Note: the container's ROS 2 setup script is at
`/opt/ros/humble/install/setup.bash`, **not** the more usual
`/opt/ros/humble/setup.bash` (confirmed in Phase 1 — this image's ROS
install only has `install/` and `log/` under `/opt/ros/humble/`, no top-level
`setup.bash`).

Only stderr output (both pre-existing in NVIDIA's original code, unrelated
to any of our changes):
- One CMake dev-warning: `Policy CMP0146 is not set: The FindCUDA module is
  removed` — harmless; `find_package(CUDA REQUIRED)` still resolves via the
  legacy `FindCUDA` module on this cmake 3.29.3, no action needed.
- ~15 `-Wunused-parameter` warnings in `src/pillarScatter.cpp` (NVIDIA's
  original TensorRT plugin code, e.g. unused `length`, `nbInputs`, `outputs`
  parameters in virtual overrides) — pre-existing, cosmetic, not touched.

No TRT-8.5-vs-8.6 version warnings were emitted by CMake/the compiler at
build time (TensorRT version compatibility, if it matters, would only
surface at engine-build time — see Step 7).

Binary produced: `/workspace/perception_ws/install/cuda_pointpillars_ros/lib/cuda_pointpillars_ros/pc_process`.
ONNX re-verified intact immediately after build (19 MB, same md5).

### Step 7 — Smoke test result: **SUCCESS (full success, beyond the minimum bar)**

First run (`ros2 run cuda_pointpillars_ros pc_process` under a 240s
`timeout`, output redirected to a log file): the process started, subscribed,
and stayed alive in `rclcpp::spin()` the entire time with 0% CPU
(confirmed via `/proc/<pid>/wchan` → `futex_wait_queue_me`, i.e. idle, not
hung). The log file appeared empty while it ran — this is **stdio full
buffering on a non-TTY**, not a hang: when `timeout` sent SIGTERM at 240s,
rclcpp's signal handler triggered a flush and the buffered `Getinfo()` output
appeared (`GPU has cuda devices: 1`, `GPU : Orin`, `Capbility: 8.7`),
confirming CUDA device detection works correctly inside the node and that
`sm_87`/`Capability 8.7` lines up with the Step 2 GENCODE fix.

Important code-reading correction to the task brief's assumption: the model
is **not** loaded and the TensorRT engine is **not** built at node startup.
`PointPillar pointpillar(Model_File, stream)` is constructed inside `infer()`,
which is only called from the subscription callback — so with no bag playing
and nothing publishing to `rslidar_points`, the engine build never starts and
the hardcoded-path fix (Step 4) is never actually exercised by simply
launching the node. The node only *looks* idle-successful; that alone is
acceptance-criterion (b) ("waiting / subscribed") but doesn't prove the model
path is correct.

To close that gap, relaunched the node (longer 600s timeout, `stdbuf -oL -eL`
for unbuffered/line-buffered log output) and used a small one-off `rclpy`
script (`/tmp/pub_test.py`, not part of the repo) to publish 5 synthetic
`PointCloud2` messages to `rslidar_points` — a single point at
`x=10, y=0, z=0, intensity=0.5` (well inside every range in `params.h`),
laid out as 4 packed float32 fields (matches the *current*, unfixed
point-parsing code's assumption — deliberately chosen so this test does not
touch or depend on the Step-3-deferred field-parsing rewrite). Confirmed via
`ros2 node info /pc_process` before publishing that the node was correctly
subscribed to `/rslidar_points` (`sensor_msgs/msg/PointCloud2`) and publishing
`/objects_marker` (`visualization_msgs/msg/MarkerArray`).

Result, captured live in the log:
```
Building TRT engine.
Enable fp16!
Bndbox objs: 0
marker size: 0
Bndbox objs: 0
marker size: 0
... (x5, one per published message)
```
- **No crash, no `exit(-1)`.** The ONNX parsed successfully from the
  Step 4 path fix.
- TensorRT engine build took ~5 minutes wall-clock on this Orin NX (process
  RSS climbed to ~3 GB during the build, CPU ~30%) — consistent with the
  task brief's "can take a few minutes" expectation, and produced
  `model/pointpillar.onnx.cache` (9.8 MB) next to the model on success.
  No TensorRT 8.5-vs-8.6 version warnings or errors appeared in the engine
  build log — TRT 8.5.2.2 built and ran this engine without complaint.
  FP16 mode was auto-enabled by the builder ("Enable fp16!").
- `Bndbox objs: 0` / `marker size: 0` (×5) is the **expected** result, not a
  bug: a single isolated point can't satisfy the model's score threshold for
  any class, so zero detections is exactly right. This run was to prove the
  pipeline executes end-to-end without crashing, not to validate detection
  quality (that depends on Step 10 / real sensor data, explicitly deferred).
- This also exercises and confirms the Step 3 MarkerArray change end-to-end:
  `infer()` returns a real `visualization_msgs::msg::MarkerArray`, the
  callback prints its size and calls `pub_->publish(marker)`, with no runtime
  errors.

Cleaned up: killed the test node, removed the one-off `/tmp/pub_test.py`
test script and YAML fixture (never committed, not part of the repo).
`model/pointpillar.onnx.cache` was left in place (it's a derived build
artifact in the README's documented location, untracked by git, harmless to
keep — re-verified `model/pointpillar.onnx` itself still 19 MB / same md5
after the entire smoke test).

### Final repo state

```
$ git status --short
 M CMakeLists.txt
 M include/cuda_pointpillars_ros/cuda_pp_ros.hpp
 M package.xml
 M src/cuda_pp_ros.cpp
 M src/pc_process.cpp
?? FIX_PLAN.md
?? model/pointpillar.onnx.cache

$ git diff --stat
 CMakeLists.txt                                | 34 ++++++++++++---------------
 include/cuda_pointpillars_ros/cuda_pp_ros.hpp |  1 -
 package.xml                                   |  1 -
 src/cuda_pp_ros.cpp                           |  2 +-
 src/pc_process.cpp                            |  1 -
 5 files changed, 16 insertions(+), 23 deletions(-)
```
Five files touched, exactly the ones anticipated in the Phase 2a plan; no
CUDA/preprocess/inference source files modified; subscription topic name
unchanged; currently on local branch `vis` (tracking `origin/vis`), not
`main` — worth deciding deliberately in a later step whether to stay on
`vis`, merge back to `main`, or cut a new branch, rather than leaving it
implicit.

### Deferred to a later phase (unchanged from §-ordering above, now confirmed still required)

- **Point-field parsing rewrite** (FIX_PLAN §3 / item 10): the node still
  blindly assumes 4 packed float32 fields. It was not exercised against the
  real bag's `x,y,z,reflectivity,tag` layout in this phase — only against a
  synthetic 4-float message shaped to match what the *current* code expects.
  This remains entirely outstanding.
- Subscription topic name `rslidar_points` — left as-is per constraints; not
  yet confirmed whether it matches the real bag's actual topic name.
- Robustness of the model-path resolution: Step 4 used the minimal hardcoded
  fix as instructed. Resolving it via
  `ament_index_cpp::get_package_share_directory` (and adding `model/` to
  `install()` in CMakeLists) is still a worthwhile follow-up, not done here.
- The git working tree is currently on a new local branch `vis`; no decision
  has been made about reconciling it with `main` going forward.

---

## PHASE_2C_CHANGES

Scope executed: the two bugs identified from Phase 2b's real-bag RViz test —
(1) point-field parsing in `infer()` blindly assumed a packed 4×float32
layout instead of reading `msg->fields`/`msg->point_step`, and (2) the
published `MarkerArray` used a hardcoded `"/rslidar"` frame with no TF link
to the cloud's actual frame (`velo_link`). **Source-only edits, done on this
x86_64 laptop with no CUDA/TensorRT — not built or run here.** All build/run
verification is deferred to the Jetson per the checklist below.

Files touched: `src/cuda_pp_ros.cpp` (main fix), `src/pc_process.cpp`
(one-line comment only). `CMakeLists.txt`, `include/cuda_pointpillars_ros/cuda_pp_ros.hpp`,
`model/pointpillar.onnx`, `package.xml` — **not touched this phase** (their
diffs in `git status` predate Phase 2c, from Phase 2a/the `vis` branch merge).

### Bug 1 fix — point-field parsing (`src/cuda_pp_ros.cpp`, `infer()`)

**Before:**
```cpp
float* points = (float*)msg->data.data();
size_t height = msg->height;
// size_t width = msg->width;
size_t row_step = msg->row_step;
size_t length = row_step * height;
size_t points_size = length/sizeof(float)/4;

float *points_data = nullptr;
unsigned int points_data_size = points_size * 4 * sizeof(float);

checkCudaErrors(cudaMallocManaged((void **)&points_data, points_data_size));
checkCudaErrors(cudaMemcpy(points_data, points, points_data_size, cudaMemcpyDefault));
checkCudaErrors(cudaDeviceSynchronize());
```

**After** (full logic, see file for the in-place comments):
```cpp
int x_offset = -1, y_offset = -1, z_offset = -1, intensity_offset = -1;
for (const auto &field : msg->fields) {
  if (field.name == "x") x_offset = field.offset;
  else if (field.name == "y") y_offset = field.offset;
  else if (field.name == "z") z_offset = field.offset;
  else if (field.name == "intensity") intensity_offset = field.offset;
}
if (intensity_offset < 0) {
  for (const auto &field : msg->fields) {
    if (field.name == "reflectivity") { intensity_offset = field.offset; break; }
  }
}
if (x_offset < 0 || y_offset < 0 || z_offset < 0 || intensity_offset < 0) {
  std::cerr << "infer(): PointCloud2 is missing x/y/z/intensity(or reflectivity) fields; dropping message." << std::endl;
  return visualization_msgs::msg::MarkerArray();
}

const size_t point_step = msg->point_step;
const size_t points_size = (point_step > 0) ? (msg->data.size() / point_step) : 0;

std::vector<float> packed_points(points_size * 4);
for (size_t i = 0; i < points_size; ++i) {
  const uint8_t *point_ptr = msg->data.data() + i * point_step;
  float x, y, z, reflectivity;
  std::memcpy(&x, point_ptr + x_offset, sizeof(float));
  std::memcpy(&y, point_ptr + y_offset, sizeof(float));
  std::memcpy(&z, point_ptr + z_offset, sizeof(float));
  std::memcpy(&reflectivity, point_ptr + intensity_offset, sizeof(float));
  packed_points[i * 4 + 0] = x;
  packed_points[i * 4 + 1] = y;
  packed_points[i * 4 + 2] = z;
  packed_points[i * 4 + 3] = reflectivity / REFLECTIVITY_SCALE;
}

float *points_data = nullptr;
unsigned int points_data_size = points_size * 4 * sizeof(float);

checkCudaErrors(cudaMallocManaged((void **)&points_data, points_data_size));
checkCudaErrors(cudaMemcpy(points_data, packed_points.data(), points_data_size, cudaMemcpyDefault));
checkCudaErrors(cudaDeviceSynchronize());
```

`REFLECTIVITY_SCALE` is a new file-scope `constexpr float` (value `255.0f`),
declared right after `Model_File` near the top of the file, with a comment
explaining why it exists. `pointpillar.doinfer(points_data, points_size, ...)`
itself is unchanged — `points_data` and `points_size` still mean exactly what
the function already expected (a packed float4-per-point buffer and a point
count), only now the buffer is genuinely packed and the count is derived from
the real per-point stride instead of an assumed 16 bytes.

**Assumptions baked into this fix:**
- x/y/z/intensity-or-reflectivity are all `FLOAT32` (datatype 7) in the
  incoming message, matching the verified bag layout in the task brief.
  Datatype is *not* re-checked per field (no `field.datatype` validation) —
  only the field *name* and *offset* are read. If a future bag encodes one of
  these fields as a different datatype (e.g. float64), this code will
  silently misread it. Flagged as an uncertainty below.
- The 4th channel is searched for as `"intensity"` first, then
  `"reflectivity"` as a fallback — matches the task brief exactly.
- `points_size` is computed as `msg->data.size() / msg->point_step`, not
  `msg->width * msg->height`. This is robust regardless of whether the driver
  sets `height`/`width`/`row_step` consistently, and degrades safely to `0`
  if `point_step` is `0` (avoids a divide-by-zero).
- Missing required fields → log to `std::cerr` and return an **empty**
  `MarkerArray` for that message, rather than crashing or reading garbage
  memory. This is new defensive behavior at the ROS message boundary; it
  does not change behavior for any well-formed message (real bag or the
  Phase 2a synthetic 4-float test message — both have all four fields).

### Bug 2 fix — frame mismatch (`src/cuda_pp_ros.cpp`, `infer()`)

**Before:**
```cpp
for (auto &pred : nms_pred) {
  obj.header.frame_id = "/rslidar";
  obj.header.stamp = msg->header.stamp;
  ...
```

**After:**
```cpp
std::string marker_frame_id = msg->header.frame_id;
if (!marker_frame_id.empty() && marker_frame_id.front() == '/') {
  marker_frame_id.erase(0, 1);
}

visualization_msgs::msg::MarkerArray objs;
visualization_msgs::msg::Marker obj;

for (auto &pred : nms_pred) {
  obj.header.frame_id = marker_frame_id;
  obj.header.stamp = msg->header.stamp;
  ...
```
`marker_frame_id` is computed once before the loop (not per-marker) and
reused for every `Marker` in the array. It is taken directly from
`msg->header.frame_id` (whatever the incoming cloud says — `velo_link` for
this bag, but this is not hardcoded), with a leading `/` stripped if present,
per ROS 2's frame-id convention (no leading slash; ROS 1 allowed it, ROS 2
does not).

### Bug 3 (small, related) — topic name comment (`src/pc_process.cpp`)

No behavior change. Added a comment immediately above the existing
`create_subscription<PointCloud2>("rslidar_points", ...)` call noting that
the bag publishes `kitti/velo` and that we remap at launch time. Subscription
name itself is untouched, per the task brief's instruction to leave it as-is.

### Build + test checklist (run on the Jetson)

```bash
# 1. Build
cd /workspace/perception_ws
source /opt/ros/humble/install/setup.bash
colcon build --packages-select cuda_pointpillars_ros

# 2. Run the node (separate terminal, after sourcing the workspace overlay)
source /opt/ros/humble/install/setup.bash
source /workspace/perception_ws/install/setup.bash
ros2 run cuda_pointpillars_ros pc_process

# 3. Play the bag — SINGLE PASS, NO --loop (69s clip; thermal safety)
#    Remap the bag's real topic name to what the node subscribes to.
ros2 bag play <path-to-bag> --remap kitti/velo:=rslidar_points

# 4. In RViz2 (separate terminal/session):
#    - Add a PointCloud2 display on the bag's native topic (kitti/velo),
#      Fixed Frame = velo_link.
#    - Add a MarkerArray display on /objects_marker.
#    - Check: MarkerArray Status should read "OK", not "Error"
#      (confirms the frame_id now matches and is resolvable — no missing-TF
#      complaint, since both displays are now in the same frame with no
#      transform needed).
#    - Check: the cube markers should sit ON/around actual point clusters
#      (cars, etc.) in the cloud, not floating off in empty space or at the
#      origin offset from everything.
#    - Watch stdout from `ros2 run` for "Bndbox objs: N" with N > 0 once the
#      bag starts publishing real frames (Phase 2b's synthetic single-point
#      test only ever produced N=0, which was expected for that test — real
#      bag frames with thousands of points should produce some detections if
#      the model/scene cooperate).
```

### Uncertain / worth verifying on the Jetson

1. **Endianness / `is_bigendian`**: `msg->is_bigendian` is not checked. x86
   and aarch64 (Orin NX) are both little-endian, and so is the producer
   recording the bag in essentially all realistic setups, so this should be
   a non-issue — but it's genuinely unverified here since this fix was
   written without running it.
2. **`point_step` padding assumption**: the task brief says `tag` is `uint8`
   at offset 16, implying `point_step` is *at least* 17, possibly padded to
   18/20/24 by the driver. The fix never assumes a specific value — it reads
   `msg->point_step` directly — but I have not seen the actual numeric value
   confirmed from the live bag's `PointCloud2` message (only the `fields`
   array contents were given in the task brief). Worth a quick
   `ros2 topic echo --no-arr /kitti/velo --field point_step` (topic name
   adjusted to whatever it resolves to) or equivalent sanity check the first
   time this runs against the real bag.
3. **Detection quality, not just position**: this phase fixes *where* boxes
   are drawn (frame) and *what data* feeds the network (parsing +
   normalization). It does not validate that `REFLECTIVITY_SCALE = 255.0f`
   is the *correct* normalization for this specific sensor's reflectivity
   range — 0–255 is the stated range in the task brief, but if the real
   sensor's driver clips/scales reflectivity differently in practice, the
   intensity channel could still be subtly wrong even though it's now in the
   right ballpark. This would show up as detections existing in roughly the
   right place but with suspect confidence/class behavior, not as a frame or
   crash issue.
4. **Performance**: the new per-point loop builds a `std::vector<float>` on
   the host and does 4 `memcpy` calls per point before the existing
   `cudaMemcpy` to device. For a typical ~100k-point LiDAR frame this is
   ~400k small memcpys — likely fine for a single 69s test clip but not
   verified for sustained real-time throughput on the Orin NX. If this turns
   out to be a bottleneck later, it's a candidate for a follow-up
   optimization (e.g. iterating once with raw offset arithmetic instead of
   `std::memcpy`, or moving the repack into a small CUDA kernel) — explicitly
   out of scope for this phase per the "no CUDA kernel changes" constraint.
5. **`x_offset`/`y_offset`/`z_offset`/`intensity_offset` are declared as
   `int`** (not `uint32_t`, which is what `sensor_msgs::msg::PointField::offset`
   actually is) so that `-1` can serve as a "not found" sentinel. This is a
   narrowing conversion but safe in practice (real point-cloud field offsets
   are always small positive numbers, nowhere near `INT_MAX`). Flagged only
   because a stricter compiler warning flag (not currently enabled in this
   project's `CMakeLists.txt`) could complain about it.

