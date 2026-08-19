// Validates the C++ port against vectors generated from the reference Python.
// See test/data/README.md; regenerate with test/gen_goldens.py.
#include "lidar_tracking/ab3dmot.hpp"
#include "lidar_tracking/kalman.hpp"
#include "lidar_tracking/metrics.hpp"
#include "lidar_tracking/track3d.hpp"
#include "lidar_tracking/types.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <string>
#include <vector>

using nlohmann::json;
using namespace lidar_tracking;

namespace
{
json load(const std::string & name)
{
  std::ifstream f(std::string(GOLDEN_DIR) + "/" + name);
  EXPECT_TRUE(f.good()) << "missing golden: " << name;
  json j;
  f >> j;
  return j;
}

// The generator feeds bbox3d2corners a float32 array, so the goldens carry
// float32 rounding; everything else is float64.
constexpr double kF32Tol = 1e-4;
constexpr double kF64Tol = 1e-9;

Box3D box_from_xyzwlhy(const std::vector<double> & v)
{
  Box3D b;
  b.x = v[0]; b.y = v[1]; b.z = v[2];
  b.w = v[3]; b.l = v[4]; b.h = v[5];
  b.ry = v[6];
  return b;
}
}  // namespace

// ── bbox3d2corners: the IVI-facing pose code ─────────────────────────────────
TEST(Goldens, Bbox3d2Corners)
{
  const json g = load("corners.json");
  ASSERT_FALSE(g.empty());
  for (size_t i = 0; i < g.size(); ++i) {
    const auto v = g[i]["box"].get<std::vector<double>>();
    Eigen::Matrix<double, 7, 1> box;
    box << v[0], v[1], v[2], v[3], v[4], v[5], v[6];
    const Corners got = bbox3d2corners(box);
    const auto want = g[i]["corners"].get<std::vector<std::vector<double>>>();
    for (int r = 0; r < 8; ++r) {
      for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(got(r, c), want[r][c], kF32Tol)
          << "case " << i << " corner " << r << " axis " << c;
      }
    }
  }
}

// ── giou_3d / iou_3d / dist_3d ───────────────────────────────────────────────
TEST(Goldens, Metrics)
{
  const json g = load("metrics.json");
  ASSERT_FALSE(g.empty());
  int overlapping = 0;
  for (size_t i = 0; i < g.size(); ++i) {
    const Box3D a = box_from_xyzwlhy(g[i]["a_xyzwlhy"].get<std::vector<double>>());
    const Box3D b = box_from_xyzwlhy(g[i]["b_xyzwlhy"].get<std::vector<double>>());
    EXPECT_NEAR(giou_3d(a, b), g[i]["giou_3d"].get<double>(), kF64Tol) << "pair " << i;
    EXPECT_NEAR(iou_3d(a, b), g[i]["iou_3d"].get<double>(), kF64Tol) << "pair " << i;
    EXPECT_NEAR(dist_3d(a, b), g[i]["dist_3d"].get<double>(), kF64Tol) << "pair " << i;
    if (g[i]["iou_3d"].get<double>() > 1e-9) { ++overlapping; }
  }
  EXPECT_GT(overlapping, 0) << "goldens must include overlapping pairs to be meaningful";
}

// ── within_range / orientation_correction branch boundaries ──────────────────
TEST(Goldens, Orientation)
{
  const json g = load("orientation.json");
  for (const auto & c : g["within_range"]) {
    EXPECT_NEAR(within_range(c["in"].get<double>()), c["out"].get<double>(), kF64Tol);
  }
  for (const auto & c : g["orientation_correction"]) {
    double pre = c["theta_pre"].get<double>();
    double obs = c["theta_obs"].get<double>();
    orientation_correction(&pre, &obs);
    EXPECT_NEAR(pre, c["out_pre"].get<double>(), kF64Tol)
      << "pre=" << c["theta_pre"] << " obs=" << c["theta_obs"];
    EXPECT_NEAR(obs, c["out_obs"].get<double>(), kF64Tol)
      << "pre=" << c["theta_pre"] << " obs=" << c["theta_obs"];
  }
}

// ── Kalman filter: filterpy defaults and the Joseph-form update ──────────────
TEST(Goldens, KalmanPredictUpdate)
{
  const json g = load("kalman.json");
  const auto init = g["init_state_xyz_theta_lwh"].get<std::vector<double>>();
  Eigen::Matrix<double, 7, 1> z0;
  for (int i = 0; i < 7; ++i) { z0(i) = init[i]; }

  KalmanFilter10x7 kf(z0);
  for (const auto & step : g["steps"]) {
    const std::string name = step["step"].get<std::string>();
    if (name.rfind("predict", 0) == 0) {
      kf.predict();
    } else if (name.rfind("update", 0) == 0) {
      const auto zv = step["z"].get<std::vector<double>>();
      Eigen::Matrix<double, 7, 1> z;
      for (int i = 0; i < 7; ++i) { z(i) = zv[i]; }
      kf.update(z);
    }
    const auto want_x = step["x"].get<std::vector<double>>();
    const auto want_p = step["P_diag"].get<std::vector<double>>();
    for (int i = 0; i < 10; ++i) {
      EXPECT_NEAR(kf.x()(i), want_x[i], 1e-9) << name << " state[" << i << "]";
      EXPECT_NEAR(kf.P()(i, i), want_p[i], 1e-9) << name << " P diag[" << i << "]";
    }
  }
}

// ── full Track3D scenarios: id assignment and lifetime ───────────────────────
namespace
{
void run_scenario(const json & sc, const std::string & name)
{
  Track3D tracker;
  const auto & inputs = sc["input"];
  const auto & expected = sc["output"];
  ASSERT_EQ(inputs.size(), expected.size()) << name;

  for (size_t f = 0; f < inputs.size(); ++f) {
    const auto boxes = inputs[f][0].get<std::vector<std::vector<double>>>();
    const auto scores = inputs[f][1].get<std::vector<double>>();
    const auto labels = inputs[f][2].get<std::vector<int>>();

    std::vector<Detection> dets;
    for (size_t i = 0; i < boxes.size(); ++i) {
      Detection d;
      d.x = boxes[i][0]; d.y = boxes[i][1]; d.z = boxes[i][2];
      d.w = boxes[i][3]; d.l = boxes[i][4]; d.h = boxes[i][5];
      d.yaw = boxes[i][6];
      d.score = scores[i];
      d.label = labels[i];
      dets.push_back(d);
    }

    const std::vector<TrackedObject> got = tracker.update(dets);
    const auto want = expected[f].get<std::vector<std::vector<double>>>();

    ASSERT_EQ(got.size(), want.size()) << name << " frame " << f << " track count";
    for (size_t i = 0; i < got.size(); ++i) {
      // Track ids and lifetime are what matter most; geometry is checked loosely
      // because the Python casts detections to float32 on the way in.
      EXPECT_EQ(got[i].track_id, static_cast<int>(want[i][9]))
        << name << " frame " << f << " row " << i << " track_id";
      EXPECT_EQ(got[i].label, static_cast<int>(want[i][8]))
        << name << " frame " << f << " row " << i << " label";
      EXPECT_NEAR(got[i].x, want[i][0], 1e-3) << name << " frame " << f << " row " << i << " x";
      EXPECT_NEAR(got[i].y, want[i][1], 1e-3) << name << " frame " << f << " row " << i << " y";
      EXPECT_NEAR(got[i].z, want[i][2], 1e-3) << name << " frame " << f << " row " << i << " z";
      EXPECT_NEAR(got[i].yaw, want[i][6], 1e-3) << name << " frame " << f << " row " << i << " yaw";
    }
  }
}
}  // namespace

TEST(Goldens, SequenceStraightLineCar) { run_scenario(load("sequences.json")["straight_line_car"], "straight_line_car"); }
TEST(Goldens, SequenceCrossingPedestrians) { run_scenario(load("sequences.json")["crossing_pedestrians"], "crossing_pedestrians"); }
TEST(Goldens, SequenceDropoutAndReturn) { run_scenario(load("sequences.json")["dropout_and_return"], "dropout_and_return"); }

// The only scenario that exercises orientation_correction inside the KF update:
// the detected heading flips by ~pi every frame, as the PointPillars direction
// classifier does in practice. Without the correction the filter fights the
// flip and the heading (and eventually the track) degrades.
TEST(Goldens, SequenceYawFlip) { run_scenario(load("sequences.json")["yaw_flip"], "yaw_flip"); }
