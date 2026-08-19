// Copyright (c) 2020 Xinshuo Weng (MIT License). See LICENSE.
#ifndef LIDAR_TRACKING__TRACK3D_HPP_
#define LIDAR_TRACKING__TRACK3D_HPP_

#include "lidar_tracking/ab3dmot.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace lidar_tracking
{

/// Class labels, in THIS pipeline's order (object_detection_msgs/Object3d.msg).
/// The TensorRT detector emits KITTI order (Car 0, Pedestrian 1, Cyclist 2) and
/// must be remapped before reaching here -- see the detector node's adapter.
enum Label : int
{
  kPedestrian = 0,
  kCyclist = 1,
  kCar = 2,
};

/// One detection going in: z is the box BOTTOM, dims follow the detector's
/// (w = along-heading, l = across, h = up) layout.
struct Detection
{
  double x{0.0}, y{0.0}, z{0.0};
  double w{0.0}, l{0.0}, h{0.0};
  double yaw{0.0};
  double score{0.0};
  int label{0};
};

/// One track coming out. Same box layout, plus a stable id.
struct TrackedObject
{
  double x{0.0}, y{0.0}, z{0.0};
  double w{0.0}, l{0.0}, h{0.0};
  double yaw{0.0};
  double score{0.0};
  int label{0};
  int track_id{0};
};

/// Multi-class 3D tracker: one AB3DMOT instance per class, sharing a single
/// global id counter so ids never collide across classes.
class Track3D
{
public:
  Track3D();

  /// Run one frame. MUST be called every frame, even with no detections, so
  /// that existing tracks age out. Detections whose label is not one of the
  /// three known classes are dropped.
  std::vector<TrackedObject> update(const std::vector<Detection> & detections);

  void reset();

  /// Override the per-class parameters (e.g. to restore the reference's
  /// bug-for-bug Cyclist threshold). Takes effect on the next reset().
  void set_params(int label, const ClassParams & params);

private:
  AB3DMOT * tracker_for(int label);

  // Insertion-ordered, mirroring the Python dict iteration order that decides
  // the order tracks are emitted in.
  std::vector<std::pair<int, std::unique_ptr<AB3DMOT>>> trackers_;
  std::vector<std::pair<int, ClassParams>> params_;
  int next_id_{1};
};

}  // namespace lidar_tracking

#endif  // LIDAR_TRACKING__TRACK3D_HPP_
