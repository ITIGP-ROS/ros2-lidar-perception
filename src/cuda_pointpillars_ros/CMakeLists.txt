cmake_minimum_required(VERSION 3.8)
project(cuda_pointpillars_ros)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-deprecated-declarations -O2")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(object_detection_msgs REQUIRED)
find_package(lidar_tracking REQUIRED)
find_package(CUDA REQUIRED)

# ── Architecture ─────────────────────────────────────────────────────────────
# The deployment target is a Jetson Orin NX (aarch64, JetPack 6.2 / L4T 36.5:
# CUDA 12.6, TensorRT 10.3). These paths differ on an x86_64 workstation, so
# detect rather than hardcode -- otherwise the package only builds on one of
# the two machines.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(CUDA_INSTALL_TARGET_DIR targets/aarch64-linux)
  set(DEFAULT_TENSORRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu/)
  set(DEFAULT_TENSORRT_LIBRARY_DIRS /usr/lib/aarch64-linux-gnu/)
  # Orin NX is sm_87 only; the original multi-arch list (30..89) was trimmed
  # for build speed.
  set(GPU_SMS "87" CACHE STRING "CUDA SM architectures to generate code for")
else()
  set(CUDA_INSTALL_TARGET_DIR targets/x86_64-linux)
  set(DEFAULT_TENSORRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu/)
  set(DEFAULT_TENSORRT_LIBRARY_DIRS /usr/lib/x86_64-linux-gnu/)
  # Ampere by default; override with -DGPU_SMS=<arch> for other dev GPUs.
  set(GPU_SMS "86" CACHE STRING "CUDA SM architectures to generate code for")
endif()

# ── Cross-compiling (Yocto/BitBake) ──────────────────────────────────────────
# The defaults above are the paths JetPack's .debs use ON A RUNNING BOARD. A
# Yocto cross build stages TensorRT somewhere else entirely: meta-tegra's
# tensorrt-core and tensorrt-plugins-prebuilt install the headers to
# ${includedir} and the libraries to ${libdir} inside the target sysroot, with
# no aarch64-linux-gnu component at all.
#
# These are CACHE variables so the recipe can redirect them with -D. They must
# NOT be plain set(): a normal variable shadows the cache for the rest of the
# directory scope, so the -D would be accepted and then silently ignored, and
# the compiler would be pointed at the BUILD HOST's /usr/include instead.
set(TENSORRT_INCLUDE_DIRS "${DEFAULT_TENSORRT_INCLUDE_DIRS}"
    CACHE PATH "Directory containing NvInfer.h")
set(TENSORRT_LIBRARY_DIRS "${DEFAULT_TENSORRT_LIBRARY_DIRS}"
    CACHE PATH "Directory containing libnvinfer.so")

# Same trap, one level up. meta-tegra's cuda.bbclass already hands
# find_package(CUDA) the right toolkit via -DCUDA_TOOLKIT_ROOT_DIR, and the
# find_package() call above consumes it correctly -- but a plain set() here
# would shadow it afterwards and send include_directories() at the host's CUDA.
# Guarded on CMAKE_CROSSCOMPILING so a native build behaves exactly as before.
if(NOT CMAKE_CROSSCOMPILING)
  set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
  set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/${CUDA_INSTALL_TARGET_DIR}/include)
endif()

# SASS for each listed arch. The trailing PTX (code=compute_XX) is a JIT
# fallback for GPUs we did not compile for; it is OFF by default because the
# deployment GPU is known (Orin NX = sm_87), it costs JIT time at startup, and
# a toolkit newer than the installed driver produces PTX the driver refuses to
# load ("unsupported toolchain", CUDA error 222). Turn it on for portability.
option(CUDA_EMBED_PTX "Embed a PTX fallback for forward compatibility" OFF)
foreach(sm ${GPU_SMS})
  set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()
if(CUDA_EMBED_PTX)
  list(GET GPU_SMS -1 HIGHEST_SM)
  set(GENCODE ${GENCODE} -gencode arch=compute_${HIGHEST_SM},code=compute_${HIGHEST_SM})
endif()

# NOTE: ${GENCODE} was previously computed and then never referenced, so nvcc
# silently fell back to its default architecture (sm_75) and the "Orin NX is
# sm_87" trimming above had no effect whatsoever. On Jetson that was masked by
# PTX JIT from sm_75 up to sm_87 -- it ran, just with slower startup and
# non-optimal code. It must actually be passed to nvcc.
# Per-stage CUDA timing. Off unless asked for: three lines of stdout per frame
# is noise. Measured cost is ~1% (21.00 vs 20.80 ms/frame), not more.
option(PERFORMANCE_LOG "Per-stage CUDA timing printouts (slows the pipeline)" OFF)
if(PERFORMANCE_LOG)
  add_compile_definitions(PERFORMANCE_LOG=1)
endif()

# The rpath is for a native build, where TensorRT sits in a multiarch directory
# the loader does not search by default. In a cross build TENSORRT_LIBRARY_DIRS
# is a path on the BUILD HOST, so baking it into the binary is both useless at
# runtime and a Yocto QA failure (bad-rpath/buildpaths) -- and unnecessary,
# because the image installs TensorRT into ${libdir}, already on the default
# loader path.
if(CMAKE_CROSSCOMPILING)
  set(TENSORRT_LINK_FLAGS -Xlinker -L${TENSORRT_LIBRARY_DIRS})
else()
  set(TENSORRT_LINK_FLAGS
      -Xlinker -rpath=${TENSORRT_LIBRARY_DIRS}
      -Xlinker -L${TENSORRT_LIBRARY_DIRS})
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    ${GENCODE}
    -ccbin ${CMAKE_CXX_COMPILER}
    -Xcompiler -DWIN_INTERFACE_CUSTOM
    ${TENSORRT_LINK_FLAGS}
)

link_directories(${TENSORRT_LIBRARY_DIRS})

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIRS}
  include
)

cuda_add_executable(pc_process
  src/pc_process.cpp
  src/cuda_pp_ros.cpp
  src/detection_adapter.cpp
  src/pillarScatter.cpp
  src/pointpillar.cpp
  src/postprocess.cpp
  src/preprocess.cpp
  src/postprocess_kernels.cu
  src/pillarScatterKernels.cu
  src/preprocess_kernels.cu
)

target_include_directories(pc_process PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIRS}
)
target_compile_features(pc_process PUBLIC c_std_99 cxx_std_17)

ament_target_dependencies(pc_process
  rclcpp
  sensor_msgs
  visualization_msgs
  geometry_msgs
  object_detection_msgs
  lidar_tracking
)

target_link_libraries(pc_process
  libnvinfer.so
  libnvonnxparser.so
)

install(TARGETS pc_process DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  # The adapter TU deliberately pulls in no TensorRT and makes no CUDA calls, so
  # the detector -> tracker boundary can be tested without a GPU.
  ament_add_gtest(test_adapter test/test_adapter.cpp src/detection_adapter.cpp)
  target_include_directories(test_adapter PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})
  ament_target_dependencies(test_adapter
    rclcpp sensor_msgs visualization_msgs geometry_msgs
    object_detection_msgs lidar_tracking)
endif()
install(DIRECTORY model/ DESTINATION share/${PROJECT_NAME}/model
        FILES_MATCHING PATTERN "*.onnx")

ament_package()
