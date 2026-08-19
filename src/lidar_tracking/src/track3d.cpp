// Copyright (c) 2020 Xinshuo Weng (MIT License). See LICENSE.
#include "lidar_tracking/track3d.hpp"

#include <algorithm>
#include <set>

namespace lidar_tracking
{

Track3D::Track3D()
{
  params_ = {
    {kPedestrian, pedestrian_params()},
    {kCyclist, cyclist_params()},
    {kCar, car_params()},
  };
}

void Track3D::set_params(int label, const ClassParams & params)
{
  for (auto & [l, p] : params_) {
    if (l == label) {
      p = params;
      return;
    }
  }
  params_.emplace_back(label, params);
}

void Track3D::reset()
{
  trackers_.clear();
  next_id_ = 1;
}

AB3DMOT * Track3D::tracker_for(int label)
{
  for (auto & [l, t] : trackers_) {
    if (l == label) {
      return t.get();
    }
  }
  const auto it = std::find_if(
    params_.begin(), params_.end(), [label](const auto & kv) {return kv.first == label;});
  if (it == params_.end()) {
    return nullptr;      // unknown class: detections are dropped, as in Python
  }
  trackers_.emplace_back(
    label, std::make_unique<AB3DMOT>(it->second, [this]() {return next_id_++;}));
  return trackers_.back().second.get();
}

std::vector<TrackedObject> Track3D::update(const std::vector<Detection> & detections)
{
  // Instantiate trackers for labels seen this frame, in ascending label order
  // (np.unique sorts), so the insertion order matches the Python's.
  std::set<int> present;
  for (const auto & d : detections) {
    present.insert(d.label);
  }
  for (const int label : present) {
    tracker_for(label);
  }

  std::vector<TrackedObject> out;
  // Every existing tracker must be advanced, including ones with no detections
  // this frame, so their tracks age out correctly.
  for (auto & [label, tracker] : trackers_) {
    std::vector<Box3D> dets;
    std::vector<std::pair<double, double>> info;
    for (const auto & d : detections) {
      if (d.label != label) {
        continue;
      }
      Eigen::Matrix<double, 7, 1> raw;
      raw << d.h, d.w, d.l, d.x, d.y, d.z, d.yaw;   // tracker.py::_LIDAR2RAW_IDX
      dets.push_back(Box3D::from_array_raw(raw));
      info.emplace_back(d.score, static_cast<double>(label));
    }

    for (const auto & r : tracker->track(dets, info)) {
      TrackedObject o;
      o.x = r.box.x;
      o.y = r.box.y;
      o.z = r.box.z;
      o.w = r.box.w;
      o.l = r.box.l;
      o.h = r.box.h;
      o.yaw = r.box.ry;
      o.score = r.score;
      o.label = static_cast<int>(r.label);
      o.track_id = r.id;
      out.push_back(o);
    }
  }
  return out;
}

}  // namespace lidar_tracking
