// Copyright (c) 2020 Xinshuo Weng (MIT License). See LICENSE.
#ifndef LIDAR_TRACKING__AB3DMOT_HPP_
#define LIDAR_TRACKING__AB3DMOT_HPP_

#include "lidar_tracking/assignment.hpp"
#include "lidar_tracking/kalman.hpp"
#include "lidar_tracking/types.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace lidar_tracking
{

/// Wrap an angle into [-pi, pi). model.py::within_range.
double within_range(double theta);

/// model.py::orientation_correction -- bring a propagated track's heading and
/// an observed heading within 90 degrees of each other before the KF update.
void orientation_correction(double * theta_pre, double * theta_obs);

struct ClassParams
{
  Algorithm algm{Algorithm::kHungarian};
  Metric metric{Metric::kGiou3d};
  double threshold{-0.2};
  int min_hits{3};
  int max_age{2};
};

/// Defaults from model.py::_PARAMS, tuned for KITTI-style lidar detections.
ClassParams car_params();
ClassParams pedestrian_params();

/// NOTE: the reference Python has `thres=2.0` here, which is a SIGN ERROR.
/// For dist_3d the affinity is `-dist3d(...)`, so it is always <= 0, and
/// data_association rejects any pair whose affinity is BELOW the threshold.
/// A positive threshold therefore rejects every candidate unconditionally:
/// cyclists never associate, a new track is born every frame, and none ever
/// survives to min_hits. Verified against the reference -- a stationary
/// cyclist yields ids [[1], [2,1], [3,2,1], [], [], ...].
///
/// This default is the corrected -2.0, which matches both upstream AB3DMOT's
/// distance thresholds and the "thresholds are NEGATIVE" comment in model.py.
/// Pass the original 2.0 if you need bug-for-bug parity with the Python.
ClassParams cyclist_params();

/// Single-class tracker: Kalman prediction + association + birth/death.
class AB3DMOT
{
public:
  struct Output
  {
    Box3D box;
    int id{0};
    double score{0.0};
    double label{0.0};
  };

  AB3DMOT(const ClassParams & params, std::function<int()> id_allocator);

  /// One frame. Must be called every frame, even with no detections, so that
  /// existing tracks age out correctly.
  std::vector<Output> track(
    const std::vector<Box3D> & dets,
    const std::vector<std::pair<double, double>> & info);   // (score, label)

private:
  struct Tracklet
  {
    KalmanFilter10x7 kf;
    int id{0};
    int hits{1};
    int time_since_update{0};
    double score{0.0};
    double label{0.0};
  };

  std::vector<Box3D> prediction();
  void update_matched(
    const Association & assoc, const std::vector<Box3D> & dets,
    const std::vector<std::pair<double, double>> & info);
  void birth(
    const std::vector<Box3D> & dets,
    const std::vector<std::pair<double, double>> & info,
    const std::vector<int> & unmatched_dets);
  std::vector<Output> collect();

  ClassParams params_;
  std::function<int()> id_allocator_;
  std::vector<std::unique_ptr<Tracklet>> trackers_;
  int frame_count_{0};
};

}  // namespace lidar_tracking

#endif  // LIDAR_TRACKING__AB3DMOT_HPP_
