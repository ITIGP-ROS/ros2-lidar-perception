// Copyright (c) 2020 Xinshuo Weng (MIT License). See LICENSE.
#include "lidar_tracking/geometry.hpp"

#include <algorithm>
#include <cmath>

namespace lidar_tracking
{

namespace
{
// dist_metrics.py::polygon_clip::inside -- strict `>`, kept strict here.
inline bool inside(const Vec2 & p, const Vec2 & cp1, const Vec2 & cp2)
{
  return (cp2.x() - cp1.x()) * (p.y() - cp1.y()) > (cp2.y() - cp1.y()) * (p.x() - cp1.x());
}

// dist_metrics.py::polygon_clip::computeIntersection
inline Vec2 intersection(const Vec2 & cp1, const Vec2 & cp2, const Vec2 & s, const Vec2 & e)
{
  const double dcx = cp1.x() - cp2.x(), dcy = cp1.y() - cp2.y();
  const double dpx = s.x() - e.x(), dpy = s.y() - e.y();
  const double n1 = cp1.x() * cp2.y() - cp1.y() * cp2.x();
  const double n2 = s.x() * e.y() - s.y() * e.x();
  const double den = dcx * dpy - dcy * dpx;
  const double n3 = 1.0 / den;
  return Vec2((n1 * dpx - n2 * dcx) * n3, (n1 * dpy - n2 * dcy) * n3);
}
}  // namespace

std::vector<Vec2> polygon_clip(const std::vector<Vec2> & subject, const std::vector<Vec2> & clip)
{
  if (subject.empty() || clip.empty()) {
    return {};
  }
  std::vector<Vec2> output = subject;
  Vec2 cp1 = clip.back();

  for (const auto & cp2 : clip) {
    const std::vector<Vec2> input = output;
    output.clear();
    Vec2 s = input.back();
    for (const auto & e : input) {
      if (inside(e, cp1, cp2)) {
        if (!inside(s, cp1, cp2)) {
          output.push_back(intersection(cp1, cp2, s, e));
        }
        output.push_back(e);
      } else if (inside(s, cp1, cp2)) {
        output.push_back(intersection(cp1, cp2, s, e));
      }
      s = e;
    }
    cp1 = cp2;
    if (output.empty()) {
      return {};   // Python returns None here
    }
  }
  return output;
}

double polygon_area(const std::vector<Vec2> & pts)
{
  const size_t n = pts.size();
  if (n < 3) {
    return 0.0;
  }
  double acc = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const Vec2 & a = pts[i];
    const Vec2 & b = pts[(i + 1) % n];
    acc += a.x() * b.y() - a.y() * b.x();
  }
  return std::abs(acc) * 0.5;
}

std::vector<Vec2> convex_hull(std::vector<Vec2> pts)
{
  const size_t n = pts.size();
  if (n < 3) {
    return pts;
  }
  std::sort(
    pts.begin(), pts.end(), [](const Vec2 & a, const Vec2 & b) {
      return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y());
    });
  pts.erase(
    std::unique(
      pts.begin(), pts.end(), [](const Vec2 & a, const Vec2 & b) {
        return a.x() == b.x() && a.y() == b.y();
      }), pts.end());
  if (pts.size() < 3) {
    return pts;
  }

  auto cross = [](const Vec2 & o, const Vec2 & a, const Vec2 & b) {
      return (a.x() - o.x()) * (b.y() - o.y()) - (a.y() - o.y()) * (b.x() - o.x());
    };

  const size_t m = pts.size();
  std::vector<Vec2> hull(2 * m);
  size_t k = 0;
  for (size_t i = 0; i < m; ++i) {                       // lower hull
    while (k >= 2 && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0) { --k; }
    hull[k++] = pts[i];
  }
  for (size_t i = m - 1, t = k + 1; i > 0; --i) {        // upper hull
    while (k >= t && cross(hull[k - 2], hull[k - 1], pts[i - 1]) <= 0) { --k; }
    hull[k++] = pts[i - 1];
  }
  hull.resize(k > 0 ? k - 1 : 0);   // last point equals the first
  return hull;
}

double convex_intersection_area(const std::vector<Vec2> & a, const std::vector<Vec2> & b)
{
  const std::vector<Vec2> inter = polygon_clip(a, b);
  if (inter.size() < 3) {
    return 0.0;
  }
  return polygon_area(convex_hull(inter));
}

double convex_area(const std::vector<Vec2> & a, const std::vector<Vec2> & b)
{
  std::vector<Vec2> all;
  all.reserve(a.size() + b.size());
  all.insert(all.end(), a.begin(), a.end());
  all.insert(all.end(), b.begin(), b.end());
  const std::vector<Vec2> hull = convex_hull(std::move(all));
  if (hull.size() < 3) {
    return 0.0;
  }
  return polygon_area(hull);
}

}  // namespace lidar_tracking
