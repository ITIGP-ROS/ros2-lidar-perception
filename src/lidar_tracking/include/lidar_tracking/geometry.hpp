// Copyright (c) 2020 Xinshuo Weng (MIT License). See LICENSE.
#ifndef LIDAR_TRACKING__GEOMETRY_HPP_
#define LIDAR_TRACKING__GEOMETRY_HPP_

#include "lidar_tracking/types.hpp"

#include <vector>

namespace lidar_tracking
{

/// Sutherland-Hodgman clip of `subject` by the CONVEX polygon `clip`.
/// Both must be counter-clockwise. Returns an empty vector where the Python
/// returns None (fully clipped away).
std::vector<Vec2> polygon_clip(const std::vector<Vec2> & subject, const std::vector<Vec2> & clip);

/// Shoelace area of an ordered polygon (absolute, so winding does not matter).
double polygon_area(const std::vector<Vec2> & pts);

/// Counter-clockwise convex hull (monotone chain).
/// Replaces scipy.spatial.ConvexHull; returns fewer than 3 points for
/// degenerate input, where scipy would instead raise QhullError.
std::vector<Vec2> convex_hull(std::vector<Vec2> pts);

/// Area of the intersection of two convex polygons.
/// Mirrors convex_hull_intersection(): scipy's `ConvexHull(...).volume` is the
/// AREA for 2D input (`.area` would be the perimeter). Degenerate results give
/// 0.0 -- the explicit decision C++ needs where scipy raises.
double convex_intersection_area(const std::vector<Vec2> & a, const std::vector<Vec2> & b);

/// Area of the convex hull of the union of both point sets (the GIoU "C" term).
double convex_area(const std::vector<Vec2> & a, const std::vector<Vec2> & b);

}  // namespace lidar_tracking

#endif  // LIDAR_TRACKING__GEOMETRY_HPP_
