// Copyright (c) 2020 Xinshuo Weng (MIT License). See LICENSE.
#ifndef LIDAR_TRACKING__METRICS_HPP_
#define LIDAR_TRACKING__METRICS_HPP_

#include "lidar_tracking/types.hpp"

namespace lidar_tracking
{

enum class Metric
{
  kGiou3d,
  kIou3d,
  kDist3d,
};

/// 3D IoU / GIoU, computed in BEV (x-y) with a separate z overlap term, exactly
/// as dist_metrics.py::iou. Only the 3D variants the tuned parameters actually
/// use are ported; iou_2d / giou_2d / m_dis / euler are dead in this pipeline.
double iou_3d(const Box3D & a, const Box3D & b);
double giou_3d(const Box3D & a, const Box3D & b);

/// Euclidean distance between the mean of the 8 corners of each box.
double dist_3d(const Box3D & a, const Box3D & b);

/// Affinity as data_association consumes it: IoU-like metrics are used
/// directly, distances are negated so that larger is always better.
double affinity(const Box3D & det, const Box3D & trk, Metric metric);

}  // namespace lidar_tracking

#endif  // LIDAR_TRACKING__METRICS_HPP_
