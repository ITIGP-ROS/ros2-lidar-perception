// Copyright (c) 2020 Xinshuo Weng (MIT License). See LICENSE.
#include "lidar_tracking/types.hpp"

#include <cmath>

namespace lidar_tracking
{

Eigen::Matrix<double, 7, 1> Box3D::to_array() const
{
  Eigen::Matrix<double, 7, 1> a;
  a << x, y, z, ry, l, w, h;
  return a;
}

Eigen::Matrix<double, 7, 1> Box3D::to_array_raw() const
{
  Eigen::Matrix<double, 7, 1> a;
  a << h, w, l, x, y, z, ry;
  return a;
}

Box3D Box3D::from_array(const Eigen::Matrix<double, 7, 1> & d)
{
  Box3D b;
  b.x = d(0); b.y = d(1); b.z = d(2); b.ry = d(3);
  b.l = d(4); b.w = d(5); b.h = d(6);
  return b;
}

Box3D Box3D::from_array_raw(const Eigen::Matrix<double, 7, 1> & d)
{
  Box3D b;
  b.h = d(0); b.w = d(1); b.l = d(2);
  b.x = d(3); b.y = d(4); b.z = d(5); b.ry = d(6);
  return b;
}

const Corners & Box3D::corners() const
{
  if (corners_valid_) {
    return corners_;
  }
  // box.py::box2corners3d -- rotz(ry) applied to a body-frame corner set where
  // x uses l (along heading), y uses w, and z runs from 0 (bottom) to h (top).
  const double c = std::cos(ry), s = std::sin(ry);
  const double xs[8] = {l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2};
  const double ys[8] = {w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2};
  const double zs[8] = {h, h, h, h, 0.0, 0.0, 0.0, 0.0};

  for (int i = 0; i < 8; ++i) {
    // R = [[c, -s, 0], [s, c, 0], [0, 0, 1]] applied as a column vector.
    corners_(i, 0) = c * xs[i] - s * ys[i] + x;
    corners_(i, 1) = s * xs[i] + c * ys[i] + y;
    corners_(i, 2) = zs[i] + z;
  }
  corners_valid_ = true;
  return corners_;
}

Corners bbox3d2corners(const Eigen::Matrix<double, 7, 1> & box)
{
  // Verbatim port of utils/process.py::bbox3d2corners. Pinned by
  // test/data/corners.json -- do not re-derive.
  //
  // The Python builds rot_mat as [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
  // with shape (3, 3, n), then np.transpose(..., (2, 1, 0)). For each element
  // that transposes the 3x3 as well, giving [[cos, -sin, 0], [sin, cos, 0],
  // [0, 0, 1]], which is then applied as a ROW-vector product `corners @ R`.
  // Row-vector times that matrix is a rotation by -yaw, which is exactly what
  // the source comment ("in fact, -angle") means.
  //
  // Here dims[0] (w) scales x, dims[1] (l) scales y, dims[2] (h) scales z, and
  // z runs 0 -> h so the stored z is the box BOTTOM.
  static const double kUnit[8][3] = {
    {-0.5, -0.5, 0.0}, {-0.5, -0.5, 1.0}, {-0.5, 0.5, 1.0}, {-0.5, 0.5, 0.0},
    {0.5, -0.5, 0.0}, {0.5, -0.5, 1.0}, {0.5, 0.5, 1.0}, {0.5, 0.5, 0.0}};

  const double cx = box(0), cy = box(1), cz = box(2);
  const double dw = box(3), dl = box(4), dh = box(5);
  const double yaw = box(6);
  const double c = std::cos(yaw), s = std::sin(yaw);

  Corners out;
  for (int i = 0; i < 8; ++i) {
    const double vx = kUnit[i][0] * dw;
    const double vy = kUnit[i][1] * dl;
    const double vz = kUnit[i][2] * dh;
    out(i, 0) = vx * c + vy * s + cx;
    out(i, 1) = -vx * s + vy * c + cy;
    out(i, 2) = vz + cz;
  }
  return out;
}

}  // namespace lidar_tracking
