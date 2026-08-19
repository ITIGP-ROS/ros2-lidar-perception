// Copyright (c) 2020 Xinshuo Weng (MIT License). See LICENSE.
#include "lidar_tracking/ab3dmot.hpp"

#include <algorithm>
#include <cmath>
#include <set>

namespace lidar_tracking
{

namespace
{
constexpr double kPi = 3.14159265358979323846;
}

double within_range(double theta)
{
  if (theta >= kPi) {
    theta -= kPi * 2;
  }
  if (theta < -kPi) {
    theta += kPi * 2;
  }
  return theta;
}

void orientation_correction(double * theta_pre, double * theta_obs)
{
  double pre = within_range(*theta_pre);
  double obs = within_range(*theta_obs);

  // If the two headings are not within an acute angle, flip the prediction.
  const double d = std::abs(obs - pre);
  if (d > kPi / 2.0 && d < kPi * 3 / 2.0) {
    pre = within_range(pre + kPi);
  }

  // Now acute (< 90) or > 270; fold the > 270 case back down to < 90.
  if (std::abs(obs - pre) >= kPi * 3 / 2.0) {
    pre += (obs > 0) ? kPi * 2 : -kPi * 2;
  }

  *theta_pre = pre;
  *theta_obs = obs;
}

ClassParams car_params()
{
  return ClassParams{Algorithm::kHungarian, Metric::kGiou3d, -0.2, 3, 2};
}

ClassParams pedestrian_params()
{
  return ClassParams{Algorithm::kGreedy, Metric::kGiou3d, -0.4, 1, 4};
}

ClassParams cyclist_params()
{
  // -2.0, not the reference's +2.0. See the header for why.
  return ClassParams{Algorithm::kHungarian, Metric::kDist3d, -2.0, 3, 4};
}

AB3DMOT::AB3DMOT(const ClassParams & params, std::function<int()> id_allocator)
: params_(params), id_allocator_(std::move(id_allocator))
{
}

std::vector<Box3D> AB3DMOT::prediction()
{
  std::vector<Box3D> trks;
  trks.reserve(trackers_.size());
  for (auto & trk : trackers_) {
    trk->kf.predict();
    trk->kf.x()(3) = within_range(trk->kf.x()(3));
    trk->time_since_update += 1;
    trks.push_back(Box3D::from_array(trk->kf.x().head<7>()));
  }
  return trks;
}

void AB3DMOT::update_matched(
  const Association & assoc, const std::vector<Box3D> & dets,
  const std::vector<std::pair<double, double>> & info)
{
  // The Python walks tracks in index order and skips those in unmatched_trks,
  // which is exactly the set of matched pairs. Each pair touches a distinct
  // tracker, so iterating the matches directly is equivalent.
  for (const auto & [d, t] : assoc.matches) {
    Tracklet & trk = *trackers_[t];
    trk.time_since_update = 0;
    trk.hits += 1;

    Eigen::Matrix<double, 7, 1> bbox3d = dets[d].to_array();
    double theta_pre = trk.kf.x()(3);
    double theta_obs = bbox3d(3);
    orientation_correction(&theta_pre, &theta_obs);
    trk.kf.x()(3) = theta_pre;
    bbox3d(3) = theta_obs;

    trk.kf.update(bbox3d);
    trk.kf.x()(3) = within_range(trk.kf.x()(3));

    trk.score = info[d].first;
    trk.label = info[d].second;
  }
}

void AB3DMOT::birth(
  const std::vector<Box3D> & dets,
  const std::vector<std::pair<double, double>> & info,
  const std::vector<int> & unmatched_dets)
{
  for (const int i : unmatched_dets) {
    auto trk = std::make_unique<Tracklet>(Tracklet{
        KalmanFilter10x7(dets[i].to_array()), id_allocator_(), 1, 0,
        info[i].first, info[i].second});
    trackers_.push_back(std::move(trk));
  }
}

std::vector<AB3DMOT::Output> AB3DMOT::collect()
{
  // model.py::output -- walk backwards so dead tracks can be erased in place,
  // and emit in that same reversed order (the Python appends while reversed).
  std::vector<Output> results;
  for (int i = static_cast<int>(trackers_.size()) - 1; i >= 0; --i) {
    Tracklet & trk = *trackers_[i];

    const bool alive = trk.time_since_update < params_.max_age;
    const bool stable = trk.hits >= params_.min_hits || frame_count_ <= params_.min_hits;
    if (alive && stable) {
      Output o;
      o.box = Box3D::from_array(trk.kf.x().head<7>());
      o.id = trk.id;
      o.score = trk.score;
      o.label = trk.label;
      results.push_back(o);
    }

    if (trk.time_since_update >= params_.max_age) {
      trackers_.erase(trackers_.begin() + i);
    }
  }
  return results;
}

std::vector<AB3DMOT::Output> AB3DMOT::track(
  const std::vector<Box3D> & dets,
  const std::vector<std::pair<double, double>> & info)
{
  frame_count_ += 1;

  const std::vector<Box3D> trks = prediction();
  const Association assoc =
    data_association(dets, trks, params_.metric, params_.threshold, params_.algm);

  update_matched(assoc, dets, info);
  birth(dets, info, assoc.unmatched_dets);
  return collect();
}

}  // namespace lidar_tracking
