// Copyright (c) 2020 Xinshuo Weng (MIT License). See LICENSE.
#include "lidar_tracking/metrics.hpp"

#include "lidar_tracking/geometry.hpp"

#include <algorithm>
#include <cmath>

namespace lidar_tracking
{

namespace
{

/// dist_metrics.py::compute_bottom -- corners[7:3:-1] i.e. rows 7, 6, 5, 4,
/// which puts the ground quad in counter-clockwise order for the clipper.
std::vector<Vec2> bottom_ccw(const Box3D & b)
{
  const Corners & c = b.corners();
  std::vector<Vec2> out;
  out.reserve(4);
  for (int i : {7, 6, 5, 4}) {
    out.emplace_back(c(i, 0), c(i, 1));
  }
  return out;
}

/// dist_metrics.py::compute_height. Corner 0 is a top corner (z + h), corner 4
/// a bottom corner (z).
double overlap_height(const Box3D & a, const Box3D & b, bool inter)
{
  const Corners & c1 = a.corners();
  const Corners & c2 = b.corners();
  double zmax, zmin;
  if (inter) {
    zmax = std::min(c1(0, 2), c2(0, 2));
    zmin = std::max(c1(4, 2), c2(4, 2));
  } else {
    zmax = std::max(c1(0, 2), c2(0, 2));
    zmin = std::min(c1(4, 2), c2(4, 2));
  }
  return std::max(0.0, zmax - zmin);
}

}  // namespace

double iou_3d(const Box3D & a, const Box3D & b)
{
  const auto ba = bottom_ccw(a), bb = bottom_ccw(b);
  const double i2d = convex_intersection_area(ba, bb);
  const double i3d = i2d * overlap_height(a, b, true);
  const double u3d = a.w * a.l * a.h + b.w * b.l * b.h - i3d;
  return i3d / u3d;
}

double giou_3d(const Box3D & a, const Box3D & b)
{
  const auto ba = bottom_ccw(a), bb = bottom_ccw(b);
  const double i2d = convex_intersection_area(ba, bb);
  const double c2d = convex_area(ba, bb);

  const double i3d = i2d * overlap_height(a, b, true);
  const double u3d = a.w * a.l * a.h + b.w * b.l * b.h - i3d;
  const double c3d = c2d * overlap_height(a, b, false);
  return i3d / u3d - (c3d - u3d) / c3d;
}

double dist_3d(const Box3D & a, const Box3D & b)
{
  const Eigen::RowVector3d ca = a.corners().colwise().mean();
  const Eigen::RowVector3d cb = b.corners().colwise().mean();
  return (ca - cb).norm();
}

double affinity(const Box3D & det, const Box3D & trk, Metric metric)
{
  switch (metric) {
    case Metric::kGiou3d:
      return giou_3d(det, trk);
    case Metric::kIou3d:
      return iou_3d(det, trk);
    case Metric::kDist3d:
      return -dist_3d(det, trk);   // matching.py negates distances
  }
  return 0.0;
}

}  // namespace lidar_tracking
