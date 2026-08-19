// The detector -> tracker boundary, isolated from the node so it can be unit
// tested without CUDA, TensorRT or a running ROS graph. Every conversion here
// fails SILENTLY when wrong -- wrong class parameters, boxes at the wrong
// height, transposed dimensions -- so it is the part most worth pinning.
#include "cuda_pointpillars_ros/cuda_pp_ros.hpp"

namespace cuda_pointpillars_ros
{

PointFieldOffsets find_point_fields(const sensor_msgs::msg::PointCloud2 & msg)
{
  PointFieldOffsets o;
  for (const auto & f : msg.fields) {
    if (f.name == "x") {o.x = static_cast<int>(f.offset);} else if (f.name == "y") {
      o.y = static_cast<int>(f.offset);
    } else if (f.name == "z") {o.z = static_cast<int>(f.offset);} else if (f.name == "intensity") {
      o.intensity = static_cast<int>(f.offset);
    }
  }
  if (o.intensity < 0) {
    for (const auto & f : msg.fields) {
      if (f.name == "reflectivity") {
        o.intensity = static_cast<int>(f.offset);
        break;
      }
    }
  }
  return o;
}

bool to_detection(
  const Bndbox & box, const std::vector<int64_t> & class_remap,
  bool convert_center_z_to_bottom, lidar_tracking::Detection * out)
{
  if (box.id < 0 || static_cast<size_t>(box.id) >= class_remap.size()) {
    return false;
  }
  out->x = box.x;
  out->y = box.y;
  // The detector decodes a CENTRE z (postprocess_kernels.cu:72). Subtracting
  // h/2 is the geometrically correct conversion to the bottom-z that
  // bbox3d2corners assumes -- but the validated Python pipeline does not do it,
  // and the IVI renders correctly without it. Off by default; see the header.
  out->z = convert_center_z_to_bottom ? (box.z - box.h / 2.0) : box.z;
  out->w = box.w;                 // along-heading extent, matches dims[0] downstream
  out->l = box.l;
  out->h = box.h;
  out->yaw = box.rt;
  out->score = box.score;
  out->label = static_cast<int>(class_remap[static_cast<size_t>(box.id)]);
  return true;
}

}  // namespace cuda_pointpillars_ros
