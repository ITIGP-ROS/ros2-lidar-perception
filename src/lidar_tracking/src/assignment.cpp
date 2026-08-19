// Copyright (c) 2020 Xinshuo Weng (MIT License). See LICENSE.
#include "lidar_tracking/assignment.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

namespace lidar_tracking
{

namespace
{
constexpr double kInf = std::numeric_limits<double>::infinity();

/// Jonker-Volgenant style O(n^2 m) assignment with potentials, for n <= m.
/// Returns row -> col, every row assigned.
std::vector<int> assign_rows_le_cols(const MatrixXd & a)
{
  const int n = static_cast<int>(a.rows());
  const int m = static_cast<int>(a.cols());

  std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
  std::vector<int> p(m + 1, 0), way(m + 1, 0);

  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(m + 1, kInf);
    std::vector<char> used(m + 1, false);
    do {
      used[j0] = true;
      const int i0 = p[j0];
      double delta = kInf;
      int j1 = 0;
      for (int j = 1; j <= m; ++j) {
        if (!used[j]) {
          const double cur = a(i0 - 1, j - 1) - u[i0] - v[j];
          if (cur < minv[j]) {
            minv[j] = cur;
            way[j] = j0;
          }
          if (minv[j] < delta) {
            delta = minv[j];
            j1 = j;
          }
        }
      }
      for (int j = 0; j <= m; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }

  std::vector<int> row_to_col(n, -1);
  for (int j = 1; j <= m; ++j) {
    if (p[j] > 0) {
      row_to_col[p[j] - 1] = j - 1;
    }
  }
  return row_to_col;
}
}  // namespace

std::vector<std::pair<int, int>> linear_sum_assignment(const MatrixXd & cost)
{
  const int rows = static_cast<int>(cost.rows());
  const int cols = static_cast<int>(cost.cols());
  std::vector<std::pair<int, int>> out;
  if (rows == 0 || cols == 0) {
    return out;
  }

  if (rows <= cols) {
    const std::vector<int> r2c = assign_rows_le_cols(cost);
    for (int r = 0; r < rows; ++r) {
      if (r2c[r] >= 0) {
        out.emplace_back(r, r2c[r]);
      }
    }
  } else {
    // More rows than columns: solve the transpose, then flip the pairs back.
    const MatrixXd t = cost.transpose();
    const std::vector<int> c2r = assign_rows_le_cols(t);
    for (int c = 0; c < cols; ++c) {
      if (c2r[c] >= 0) {
        out.emplace_back(c2r[c], c);
      }
    }
    std::sort(out.begin(), out.end());   // scipy orders by ascending row
  }
  return out;
}

std::vector<std::pair<int, int>> greedy_matching(const MatrixXd & cost)
{
  const int num_dets = static_cast<int>(cost.rows());
  const int num_trks = static_cast<int>(cost.cols());
  std::vector<std::pair<int, int>> out;
  if (num_dets == 0 || num_trks == 0) {
    return out;
  }

  std::vector<int> order(static_cast<size_t>(num_dets) * num_trks);
  std::iota(order.begin(), order.end(), 0);
  // np.argsort defaults to an unstable quicksort, so the Python is already
  // order-unstable on exact ties; stable_sort here just makes us deterministic.
  std::stable_sort(
    order.begin(), order.end(), [&](int lhs, int rhs) {
      return cost(lhs / num_trks, lhs % num_trks) < cost(rhs / num_trks, rhs % num_trks);
    });

  std::vector<char> det_taken(num_dets, false), trk_taken(num_trks, false);
  for (const int idx : order) {
    const int d = idx / num_trks;
    const int t = idx % num_trks;
    if (!det_taken[d] && !trk_taken[t]) {
      det_taken[d] = true;
      trk_taken[t] = true;
      out.emplace_back(d, t);
    }
  }
  return out;
}

Association data_association(
  const std::vector<Box3D> & dets, const std::vector<Box3D> & trks,
  Metric metric, double threshold, Algorithm algm)
{
  Association result;
  const int nd = static_cast<int>(dets.size());
  const int nt = static_cast<int>(trks.size());

  if (nt == 0) {
    result.unmatched_dets.resize(nd);
    std::iota(result.unmatched_dets.begin(), result.unmatched_dets.end(), 0);
    return result;
  }
  if (nd == 0) {
    result.unmatched_trks.resize(nt);
    std::iota(result.unmatched_trks.begin(), result.unmatched_trks.end(), 0);
    return result;
  }

  MatrixXd aff(nd, nt);
  for (int d = 0; d < nd; ++d) {
    for (int t = 0; t < nt; ++t) {
      aff(d, t) = affinity(dets[d], trks[t], metric);
    }
  }

  const MatrixXd cost = -aff;
  const std::vector<std::pair<int, int>> candidates =
    (algm == Algorithm::kHungarian) ? linear_sum_assignment(cost) : greedy_matching(cost);

  std::vector<char> det_matched(nd, false), trk_matched(nt, false);
  for (const auto & [d, t] : candidates) {
    det_matched[d] = true;
    trk_matched[t] = true;
  }
  for (int d = 0; d < nd; ++d) {
    if (!det_matched[d]) {
      result.unmatched_dets.push_back(d);
    }
  }
  for (int t = 0; t < nt; ++t) {
    if (!trk_matched[t]) {
      result.unmatched_trks.push_back(t);
    }
  }

  // Reject weak pairs; both members go back to the unmatched lists. Order here
  // matters -- it decides the order new tracks are born in, hence their IDs.
  for (const auto & [d, t] : candidates) {
    if (aff(d, t) < threshold) {
      result.unmatched_dets.push_back(d);
      result.unmatched_trks.push_back(t);
    } else {
      result.matches.emplace_back(d, t);
    }
  }
  return result;
}

}  // namespace lidar_tracking
