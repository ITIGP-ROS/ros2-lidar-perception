// Copyright (c) 2020 Xinshuo Weng (MIT License). See LICENSE.
#ifndef LIDAR_TRACKING__ASSIGNMENT_HPP_
#define LIDAR_TRACKING__ASSIGNMENT_HPP_

#include "lidar_tracking/metrics.hpp"
#include "lidar_tracking/types.hpp"

#include <Eigen/Dense>
#include <utility>
#include <vector>

namespace lidar_tracking
{

using MatrixXd = Eigen::MatrixXd;

enum class Algorithm
{
  kHungarian,
  kGreedy,
};

struct Association
{
  std::vector<std::pair<int, int>> matches;   ///< (detection index, track index)
  std::vector<int> unmatched_dets;
  std::vector<int> unmatched_trks;
};

/// Minimum-cost assignment, equivalent to scipy.optimize.linear_sum_assignment.
///
/// Handles RECTANGULAR cost matrices natively (assigning min(rows, cols) pairs)
/// rather than padding to square -- padding silently changes which pairs win.
/// Returned pairs are ordered by ascending row index, as scipy does.
std::vector<std::pair<int, int>> linear_sum_assignment(const MatrixXd & cost);

/// matching.py::greedy_matching -- sort every cell by cost ascending and take
/// pairs first-come-first-served.
std::vector<std::pair<int, int>> greedy_matching(const MatrixXd & cost);

/// matching.py::data_association. `threshold` is compared against the raw
/// affinity (larger is better), so pairs BELOW it are rejected and their two
/// members pushed back into the unmatched lists.
Association data_association(
  const std::vector<Box3D> & dets, const std::vector<Box3D> & trks,
  Metric metric, double threshold, Algorithm algm);

}  // namespace lidar_tracking

#endif  // LIDAR_TRACKING__ASSIGNMENT_HPP_
