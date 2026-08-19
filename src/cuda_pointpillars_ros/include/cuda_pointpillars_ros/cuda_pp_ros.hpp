#ifndef CUDA_POINTPILLARS_ROS__CUDA_PP_ROS_HPP_
#define CUDA_POINTPILLARS_ROS__CUDA_PP_ROS_HPP_

#include "lidar_tracking/track3d.hpp"
#include "postprocess.h"          // Bndbox

#include <object_detection_msgs/msg/object3d_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <cuda_runtime.h>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

class PointPillar;   // include/pointpillar.h

namespace cuda_pointpillars_ros
{

void Getinfo();

/// Byte offsets of the four fields we need inside a PointCloud2.
struct PointFieldOffsets
{
  int x{-1}, y{-1}, z{-1}, intensity{-1};
  bool valid() const { return x >= 0 && y >= 0 && z >= 0 && intensity >= 0; }
};

/// Locate x/y/z and the intensity channel by NAME, never by assuming an order
/// or a fixed stride. RoboSense/Livox-style drivers call the 4th channel
/// "reflectivity" rather than "intensity", so both are accepted.
PointFieldOffsets find_point_fields(const sensor_msgs::msg::PointCloud2 & msg);

/// THE detector -> tracker boundary. Three conversions happen here, and all
/// three fail silently rather than loudly if they are wrong:
///
///  1. CLASS ID. The detector emits KITTI order (Car 0, Pedestrian 1,
///     Cyclist 2); this pipeline uses Pedestrian 0, Cyclist 1, Car 2. The
///     default remap is therefore {2, 0, 1}. Get this wrong and cars are
///     tracked with pedestrian parameters -- it runs, it just tracks badly.
///  2. Z ORIGIN. postprocess_kernels.cu decodes a box CENTRE z, while
///     bbox3d2corners builds corners over [z, z+h] i.e. treats z as the BOTTOM.
///     The adapter must therefore subtract h/2, and the default is ON.
///     The Python pipeline does the same subtraction -- it is just not in the
///     decode loop where you would look for it, but at the end of
///     model/anchors.py::anchors2bboxes (`z = z - h / 2`), so the z that
///     reaches its bbox3d2corners is already a bottom. Leaving this off makes
///     every box render exactly h/2 too high.
///  3. DIMENSIONS. Bndbox.w is the along-heading extent (3.9 for the KITTI Car
///     anchor), which is what bbox3d2corners' dims[0] means too, so w/l/h copy
///     straight across with NO swap. Verified in lidar_tracking's corner
///     goldens; see src/lidar_tracking/test/data/README.md.
///
/// Returns false if the class id is outside the remap table.
bool to_detection(
  const Bndbox & box, const std::vector<int64_t> & class_remap,
  bool convert_center_z_to_bottom, lidar_tracking::Detection * out);

class PointPillarsNode : public rclcpp::Node
{
public:
  explicit PointPillarsNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~PointPillarsNode() override;

private:
  void on_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  /// Repack an arbitrary PointCloud2 into the tightly packed float32 x4 buffer
  /// (x, y, z, intensity/scale, stride 16 B) that preprocess_kernels.cu needs.
  /// Returns the number of points written, reusing the host buffer each frame.
  size_t repack_points(const sensor_msgs::msg::PointCloud2 & msg);

  /// Cross-check the loaded engine against params.h and throw if they disagree.
  /// Every mismatch found during bring-up failed SILENTLY -- a model exported
  /// with different voxel/class counts produced confident, plausible-looking
  /// detections that were pure noise. Fail loudly at startup instead.
  void validate_engine_against_params();

  void publish_markers(
    const std::vector<lidar_tracking::TrackedObject> & tracks,
    const std_msgs::msg::Header & header);

  // Constructed ONCE. The original code built PointPillar (and therefore
  // deserialised the TensorRT engine) inside the per-message callback, which
  // dominated frame time by orders of magnitude on Jetson.
  std::unique_ptr<PointPillar> pointpillar_;
  cudaStream_t stream_{nullptr};

  float * points_dev_{nullptr};        ///< reused device buffer
  size_t points_capacity_{0};          ///< in points, not bytes
  std::vector<float> packed_host_;     ///< reused host staging buffer
  std::vector<Bndbox> detections_;     ///< reused output vector

  lidar_tracking::Track3D tracker_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<object_detection_msgs::msg::Object3dArray>::SharedPtr pub_objects_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;

  std::string output_frame_id_;
  std::vector<int64_t> class_remap_;
  double intensity_scale_{255.0};
  size_t max_points_{300000};
  bool publish_markers_{false};
  bool use_msg_stamp_{true};
  /// See to_detection(). Default true; false renders every box h/2 too high.
  bool convert_center_z_to_bottom_{true};

  /// Runtime cut-off on what reaches /object_detections_3d, settable with
  /// `ros2 param set`. Applied to detections BEFORE tracking, so it means
  /// exactly what raising the compiled SCORE_THRESH means -- weak boxes never
  /// reach the tracker and cannot spawn tracks. It can only ever RAISE the
  /// compiled threshold: boxes below SCORE_THRESH were discarded on the GPU and
  /// no longer exist. Atomic because the parameter callback runs on a different
  /// thread from the subscription.
  std::atomic<double> score_threshold_{0.0};

  /// Store a new runtime threshold, warning if it is below the compiled
  /// SCORE_THRESH (where it would have no effect) .
  void set_score_threshold(double v);
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_;
};

}  // namespace cuda_pointpillars_ros

#endif  // CUDA_POINTPILLARS_ROS__CUDA_PP_ROS_HPP_
