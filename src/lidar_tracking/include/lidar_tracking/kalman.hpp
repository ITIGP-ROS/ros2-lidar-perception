// Copyright (c) 2020 Xinshuo Weng (MIT License). See LICENSE.
#ifndef LIDAR_TRACKING__KALMAN_HPP_
#define LIDAR_TRACKING__KALMAN_HPP_

#include <Eigen/Dense>

namespace lidar_tracking
{

/// Constant-velocity Kalman filter matching filterpy's KalmanFilter(10, 7) as
/// configured in reference/.../tracking/kalman_filter.py.
///
/// State (10): x, y, z, theta, l, w, h, dx, dy, dz
/// Measurement (7): x, y, z, theta, l, w, h
///
/// The initial covariances come from filterpy defaults mutated as in the
/// reference: P starts as I, P[7:,7:] *= 1000 and P *= 10, landing on
/// diag([10]*7 + [10000]*3). Pinned by test/data/kalman.json.
///
/// The covariance update uses the Joseph form because filterpy does. Note this
/// is algebraically identical to the naive (I-KH)P at the OPTIMAL gain -- it
/// buys numerical conditioning, not different numbers, so a test cannot tell
/// the two apart.
class KalmanFilter10x7
{
public:
  using State = Eigen::Matrix<double, 10, 1>;
  using Meas = Eigen::Matrix<double, 7, 1>;
  using StateCov = Eigen::Matrix<double, 10, 10>;
  using MeasCov = Eigen::Matrix<double, 7, 7>;

  explicit KalmanFilter10x7(const Meas & initial);

  void predict();
  /// filterpy uses the Joseph form for the covariance update; so do we.
  void update(const Meas & z);

  State & x() { return x_; }
  const State & x() const { return x_; }
  const StateCov & P() const { return P_; }

private:
  State x_;
  StateCov P_;
  StateCov F_;
  StateCov Q_;
  Eigen::Matrix<double, 7, 10> H_;
  MeasCov R_;
};

}  // namespace lidar_tracking

#endif  // LIDAR_TRACKING__KALMAN_HPP_
