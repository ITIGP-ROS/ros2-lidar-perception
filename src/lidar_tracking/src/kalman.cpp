// Copyright (c) 2020 Xinshuo Weng (MIT License). See LICENSE.
#include "lidar_tracking/kalman.hpp"

namespace lidar_tracking
{

KalmanFilter10x7::KalmanFilter10x7(const Meas & initial)
{
  // Constant velocity: x' = x + dx, y' = y + dy, z' = z + dz; the rest hold.
  F_ = StateCov::Identity();
  F_(0, 7) = 1.0;
  F_(1, 8) = 1.0;
  F_(2, 9) = 1.0;

  // The first 7 state dimensions are observed directly.
  H_.setZero();
  for (int i = 0; i < 7; ++i) {
    H_(i, i) = 1.0;
  }

  // filterpy defaults, then the mutations from kalman_filter.py: P[7:,7:] *= 1000
  // followed by P *= 10, landing on diag([10]*7 + [10000]*3). (The two scalar
  // multiplies commute, so the order they appear in is not load-bearing -- the
  // resulting values are.) Pinned by test/data/kalman.json.
  P_ = StateCov::Identity();
  P_.bottomRightCorner<3, 3>() *= 1000.0;
  P_ *= 10.0;

  // Process noise: identity, with the constant-velocity part made more certain.
  Q_ = StateCov::Identity();
  Q_.bottomRightCorner<3, 3>() *= 0.01;

  // Measurement noise stays at filterpy's default identity -- the `R *= 10`
  // line in kalman_filter.py is commented out upstream.
  R_ = MeasCov::Identity();

  x_.setZero();
  x_.head<7>() = initial;
}

void KalmanFilter10x7::predict()
{
  // filterpy's alpha defaults to 1, so the fading-memory term is a no-op.
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter10x7::update(const Meas & z)
{
  const Eigen::Matrix<double, 7, 1> y = z - H_ * x_;
  const Eigen::Matrix<double, 10, 7> PHT = P_ * H_.transpose();
  const MeasCov S = H_ * PHT + R_;
  const Eigen::Matrix<double, 10, 7> K = PHT * S.inverse();

  x_ = x_ + K * y;

  // Joseph form, as filterpy uses. Equivalent to (I-KH)P at the optimal gain;
  // kept for its better numerical conditioning.
  const StateCov I_KH = StateCov::Identity() - K * H_;
  P_ = I_KH * P_ * I_KH.transpose() + K * R_ * K.transpose();
}

}  // namespace lidar_tracking
