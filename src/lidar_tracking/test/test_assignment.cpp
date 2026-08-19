// Assignment tests. The rectangular cases are the point: scipy's
// linear_sum_assignment handles non-square natively, and padding to square
// silently changes which pairs win.
#include "lidar_tracking/assignment.hpp"
#include "lidar_tracking/track3d.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

using namespace lidar_tracking;

namespace
{
double cost_of(const MatrixXd & c, const std::vector<std::pair<int, int>> & pairs)
{
  double total = 0.0;
  for (const auto & [r, k] : pairs) { total += c(r, k); }
  return total;
}

/// Brute-force optimum over all assignments of min(rows, cols) pairs.
double brute_force_min(const MatrixXd & c)
{
  const int rows = static_cast<int>(c.rows()), cols = static_cast<int>(c.cols());
  std::vector<int> perm(cols);
  std::iota(perm.begin(), perm.end(), 0);
  double best = std::numeric_limits<double>::infinity();
  std::sort(perm.begin(), perm.end());
  do {
    double total = 0.0;
    for (int r = 0; r < std::min(rows, cols); ++r) { total += c(r, perm[r]); }
    best = std::min(best, total);
  } while (std::next_permutation(perm.begin(), perm.end()));
  return best;
}
}  // namespace

TEST(Assignment, SquareMatchesBruteForce)
{
  MatrixXd c(3, 3);
  c << 4, 1, 3,
    2, 0, 5,
    3, 2, 2;
  const auto got = linear_sum_assignment(c);
  EXPECT_EQ(got.size(), 3u);
  EXPECT_NEAR(cost_of(c, got), brute_force_min(c), 1e-12);
}

TEST(Assignment, MoreColumnsThanRows)
{
  MatrixXd c(2, 5);
  c << 9, 1, 8, 7, 6,
    4, 3, 2, 9, 9;
  const auto got = linear_sum_assignment(c);
  ASSERT_EQ(got.size(), 2u) << "must assign every row when cols > rows";
  EXPECT_NEAR(cost_of(c, got), brute_force_min(c), 1e-12);
}

TEST(Assignment, MoreRowsThanColumns)
{
  MatrixXd c(5, 2);
  c << 9, 4,
    1, 3,
    8, 2,
    7, 9,
    6, 9;
  const auto got = linear_sum_assignment(c);
  ASSERT_EQ(got.size(), 2u) << "must assign every column when rows > cols";
  const MatrixXd t = c.transpose();
  EXPECT_NEAR(cost_of(c, got), brute_force_min(t), 1e-12);
  // scipy returns pairs ordered by ascending row index.
  EXPECT_TRUE(std::is_sorted(got.begin(), got.end()));
}

TEST(Assignment, RandomRectangularMatchesBruteForce)
{
  std::mt19937 rng(1234);
  std::uniform_real_distribution<double> d(-5.0, 5.0);
  for (int trial = 0; trial < 60; ++trial) {
    const int rows = 1 + trial % 5;
    const int cols = 1 + (trial / 5) % 6;
    MatrixXd c(rows, cols);
    for (int r = 0; r < rows; ++r) {
      for (int k = 0; k < cols; ++k) { c(r, k) = d(rng); }
    }
    const auto got = linear_sum_assignment(c);
    EXPECT_EQ(static_cast<int>(got.size()), std::min(rows, cols));
    const MatrixXd oriented = (rows <= cols) ? c : MatrixXd(c.transpose());
    EXPECT_NEAR(cost_of(c, got), brute_force_min(oriented), 1e-9)
      << "trial " << trial << " (" << rows << "x" << cols << ")";
  }
}

TEST(Assignment, EmptyInputs)
{
  EXPECT_TRUE(linear_sum_assignment(MatrixXd(0, 3)).empty());
  EXPECT_TRUE(linear_sum_assignment(MatrixXd(3, 0)).empty());
  EXPECT_TRUE(greedy_matching(MatrixXd(0, 0)).empty());
}

TEST(Assignment, GreedyIsFirstComeFirstServed)
{
  // Cheapest cell is (1,0); it wins, so row 0 must take the next best free col.
  MatrixXd c(2, 2);
  c << 5, 2,
    1, 3;
  const auto got = greedy_matching(c);
  ASSERT_EQ(got.size(), 2u);
  EXPECT_EQ(got[0], std::make_pair(1, 0));
  EXPECT_EQ(got[1], std::make_pair(0, 1));
}

// ── regression: the reference's Cyclist threshold sign error ─────────────────
// model.py has thres=+2.0 for Cyclist, but dist_3d affinity is -dist3d(...),
// always <= 0, and pairs BELOW the threshold are rejected -- so every cyclist
// association is rejected and a new id is born every frame. We default to the
// corrected -2.0; this test pins that a cyclist keeps one stable id.
TEST(CyclistThreshold, StationaryCyclistKeepsOneId)
{
  Track3D tracker;
  for (int frame = 0; frame < 8; ++frame) {
    Detection d;
    d.x = 10.0 + 0.3 * frame; d.y = 0.0; d.z = -1.7;
    d.w = 1.76; d.l = 0.6; d.h = 1.73;
    d.yaw = 0.0; d.score = 0.9; d.label = kCyclist;

    const auto tracks = tracker.update({d});
    ASSERT_EQ(tracks.size(), 1u) << "frame " << frame << ": association failed, track duplicated";
    EXPECT_EQ(tracks[0].track_id, 1) << "frame " << frame << ": id must stay stable";
  }
}

TEST(CyclistThreshold, ReferenceBugReproducibleOnRequest)
{
  // Restoring the Python's +2.0 must reproduce the broken behaviour, so the
  // bug-for-bug escape hatch is real and not silently ignored.
  Track3D tracker;
  ClassParams broken = cyclist_params();
  broken.threshold = 2.0;
  tracker.set_params(kCyclist, broken);
  tracker.reset();

  Detection d;
  d.x = 10.0; d.y = 0.0; d.z = -1.7;
  d.w = 1.76; d.l = 0.6; d.h = 1.73;
  d.yaw = 0.0; d.score = 0.9; d.label = kCyclist;

  const auto f0 = tracker.update({d});
  const auto f1 = tracker.update({d});
  EXPECT_EQ(f0.size(), 1u);
  EXPECT_EQ(f1.size(), 2u) << "with thres=+2.0 the track must duplicate";
}
