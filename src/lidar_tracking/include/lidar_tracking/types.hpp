// Copyright (c) 2020 Xinshuo Weng (MIT License) -- original AB3DMOT implementation.
// https://github.com/xinshuoweng/AB3DMOT
// C++ port of the adapted Python version kept in reference/.
#ifndef LIDAR_TRACKING__TYPES_HPP_
#define LIDAR_TRACKING__TYPES_HPP_

#include <Eigen/Dense>
#include <array>
#include <limits>
#include <vector>

namespace lidar_tracking
{

using Vec2 = Eigen::Vector2d;
using Corners = Eigen::Matrix<double, 8, 3>;

/// A 3D box in the LIDAR frame (x forward, y left, z up).
///
/// (x, y, z) is the BOTTOM centre of the box, matching the PointPillars lidar
/// convention and `reference/.../tracking/box.py`.
///
/// NOTE on w/l: this struct follows box.py, where `l` is the extent along the
/// heading and `w` is across it. The IVI-facing `bbox3d2corners()` uses the
/// OPPOSITE convention. That disagreement is pre-existing behaviour in the
/// working system -- it is applied identically to detections and tracks so
/// association stays self-consistent, and the tuned thresholds in ab3dmot.cpp
/// were validated against it. Do not "fix" it here; see reference/README.md.
struct Box3D
{
  double x{0.0}, y{0.0}, z{0.0};
  double h{0.0}, w{0.0}, l{0.0};
  double ry{0.0};
  double s{std::numeric_limits<double>::quiet_NaN()};   // score; NaN = unset

  bool has_score() const { return !std::isnan(s); }

  /// [x, y, z, ry, l, w, h] -- the Kalman filter's measurement layout.
  Eigen::Matrix<double, 7, 1> to_array() const;
  /// [h, w, l, x, y, z, ry] -- the KITTI "raw" layout.
  Eigen::Matrix<double, 7, 1> to_array_raw() const;

  static Box3D from_array(const Eigen::Matrix<double, 7, 1> & d);        // [x,y,z,ry,l,w,h]
  static Box3D from_array_raw(const Eigen::Matrix<double, 7, 1> & d);    // [h,w,l,x,y,z,ry]

  /// 8 corners, box.py ordering: 0-3 top (z + h), 4-7 bottom (z).
  const Corners & corners() const;

private:
  mutable Corners corners_;
  mutable bool corners_valid_{false};
};

/// The IVI-facing corner generation, ported verbatim from
/// `reference/.../utils/process.py::bbox3d2corners`.
///
/// Input is [x, y, z, w, l, h, yaw] with z the box BOTTOM. Here `dims[0]` (w)
/// is the extent along x/heading -- the opposite of Box3D above, deliberately.
/// This is the function that makes the poses the IVI consumes correct; it is
/// pinned by test/data/corners.json. Do not re-derive it from the maths.
Corners bbox3d2corners(const Eigen::Matrix<double, 7, 1> & box);

}  // namespace lidar_tracking

#endif  // LIDAR_TRACKING__TYPES_HPP_
