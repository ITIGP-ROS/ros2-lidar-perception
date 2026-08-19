#include "cuda_pointpillars_ros/cuda_pp_ros.hpp"

#include "lidar_tracking/types.hpp"
#include "params.h"
#include "pointpillar.h"

#include <geometry_msgs/msg/point.hpp>

#include <chrono>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace cuda_pointpillars_ros
{

void Getinfo()
{
  cudaDeviceProp prop;
  int count = 0;
  cudaGetDeviceCount(&count);
  printf("\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);
    printf("----device id: %d info----\n", i);
    printf("  GPU : %s \n", prop.name);
    printf("  Capbility: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
    printf("  Const memory: %luKB\n", prop.totalConstMem >> 10);
    printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
    printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1],
      prop.maxThreadsDim[2]);
    printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1],
      prop.maxGridSize[2]);
  }
  printf("\n");
}

PointPillarsNode::PointPillarsNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("cuda_pointpillars_node", options)
{
  const std::string model_path = declare_parameter<std::string>("model_path", "");
  const std::string input_topic = declare_parameter<std::string>("input_topic", "rslidar_points");
  const std::string output_topic =
    declare_parameter<std::string>("output_topic", "object_detections_3d");
  const std::string markers_topic =
    declare_parameter<std::string>("markers_topic", "objects_marker");
  // Empty means "inherit the incoming cloud's frame". Defaults to velo_link to
  // match the configuration the IVI was validated against.
  output_frame_id_ = declare_parameter<std::string>("output_frame_id", "velo_link");
  // Detector KITTI order {Car, Pedestrian, Cyclist} -> this pipeline's
  // {Pedestrian 0, Cyclist 1, Car 2}. A parameter because a Livox-trained
  // model may declare CLASS_NAMES in a different order.
  class_remap_ = declare_parameter<std::vector<int64_t>>("class_remap", {2, 0, 1});
  intensity_scale_ = declare_parameter<double>("intensity_scale", 255.0);
  max_points_ = static_cast<size_t>(declare_parameter<int>("max_points", 300000));
  publish_markers_ = declare_parameter<bool>("publish_markers", false);
  // The Python stamped output with the wall clock; using the cloud's stamp is
  // more correct for downstream synchronisation. Flip if the IVI disagrees.
  use_msg_stamp_ = declare_parameter<bool>("use_msg_stamp", true);
  // The decode emits a CENTRE z, bbox3d2corners wants a BOTTOM z. On by
  // default; off makes every box render h/2 too high (see to_detection()).
  convert_center_z_to_bottom_ = declare_parameter<bool>("convert_center_z_to_bottom", true);
  // Runtime cut-off for /object_detections_3d. 0.0 = publish whatever cleared
  // the compiled SCORE_THRESH.
  set_score_threshold(declare_parameter<double>("score_threshold", 0.0));
  param_cb_ = add_on_set_parameters_callback(
    [this](const std::vector<rclcpp::Parameter> & params) {
      rcl_interfaces::msg::SetParametersResult result;
      result.successful = true;
      for (const auto & p : params) {
        if (p.get_name() != "score_threshold") {
          continue;
        }
        const double v = p.as_double();
        if (v < 0.0 || v > 1.0) {
          result.successful = false;
          result.reason = "score_threshold must be in [0, 1]";
          return result;
        }
        set_score_threshold(v);
      }
      return result;
    });

  if (model_path.empty()) {
    RCLCPP_FATAL(get_logger(), "parameter 'model_path' is required (path to the .onnx)");
    throw std::runtime_error("model_path not set");
  }
  // Check the file here rather than letting the ONNX parser fail 40 lines into
  // its own log with a stat() assertion. The usual cause is a placeholder path
  // pasted verbatim out of the docs.
  {
    std::ifstream probe(model_path, std::ios::binary);
    if (!probe.good()) {
      RCLCPP_FATAL(get_logger(), "model_path does not exist or is not readable: '%s'",
                   model_path.c_str());
      throw std::runtime_error("model_path not readable: " + model_path);
    }
  }

  checkCudaErrors(cudaStreamCreate(&stream_));
  RCLCPP_INFO(get_logger(), "loading PointPillars model: %s", model_path.c_str());
  // Built once, here -- not per frame. TRT::TRT caches the serialised engine
  // next to the .onnx, so only the first ever run pays the build cost.
  pointpillar_ = std::make_unique<PointPillar>(model_path, stream_);
  validate_engine_against_params();
  RCLCPP_INFO(get_logger(), "model ready");

  packed_host_.reserve(max_points_ * 4);
  detections_.reserve(100);

  // Matches the QoS the Python detector used (commit ffea036), which is what
  // the recorded bags and the IVI expect.
  const rclcpp::QoS qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
  sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    input_topic, qos,
    std::bind(&PointPillarsNode::on_cloud, this, std::placeholders::_1));
  pub_objects_ =
    create_publisher<object_detection_msgs::msg::Object3dArray>(output_topic, qos);
  if (publish_markers_) {
    pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>(markers_topic, 10);
  }
}

PointPillarsNode::~PointPillarsNode()
{
  pointpillar_.reset();
  if (points_dev_ != nullptr) {
    cudaFree(points_dev_);
  }
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
  }
}

void PointPillarsNode::set_score_threshold(double v)
{
  const float compiled = Params().score_thresh;
  if (v > 0.0 && v < compiled) {
    RCLCPP_WARN(
      get_logger(),
      "score_threshold %.3f is below the compiled SCORE_THRESH %.3f and will have no "
      "effect -- boxes under %.3f were discarded on the GPU and cannot be recovered. "
      "Lower SCORE_THRESH in config/*.yaml and rebuild if you need them.",
      v, compiled, compiled);
  }
  score_threshold_.store(v, std::memory_order_relaxed);
  RCLCPP_INFO(get_logger(), "score_threshold = %.3f (compiled SCORE_THRESH %.3f)", v, compiled);
}

size_t PointPillarsNode::repack_points(const sensor_msgs::msg::PointCloud2 & msg)
{
  const PointFieldOffsets off = find_point_fields(msg);
  if (!off.valid()) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "PointCloud2 lacks x/y/z/intensity(or reflectivity); dropping message");
    return 0;
  }

  const size_t step = msg.point_step;
  size_t count = (step > 0) ? (msg.data.size() / step) : 0;
  if (count > max_points_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "cloud has %zu points, truncating to max_points=%zu", count, max_points_);
    count = max_points_;
  }

  packed_host_.resize(count * 4);
  const float inv_scale = (intensity_scale_ != 0.0) ?
    static_cast<float>(1.0 / intensity_scale_) : 1.0f;
  for (size_t i = 0; i < count; ++i) {
    const uint8_t * p = msg.data.data() + i * step;
    float x, y, z, intensity;
    std::memcpy(&x, p + off.x, sizeof(float));
    std::memcpy(&y, p + off.y, sizeof(float));
    std::memcpy(&z, p + off.z, sizeof(float));
    std::memcpy(&intensity, p + off.intensity, sizeof(float));
    packed_host_[i * 4 + 0] = x;
    packed_host_[i * 4 + 1] = y;
    packed_host_[i * 4 + 2] = z;
    packed_host_[i * 4 + 3] = intensity * inv_scale;
  }
  return count;
}

void PointPillarsNode::on_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  const auto t0 = std::chrono::steady_clock::now();
  const size_t count = repack_points(*msg);
  const auto t1 = std::chrono::steady_clock::now();

  detections_.clear();
  if (count > 0) {
    // Grow the device buffer only when a cloud needs more room than before.
    if (count > points_capacity_) {
      if (points_dev_ != nullptr) {
        checkCudaErrors(cudaFree(points_dev_));
      }
      checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&points_dev_), count * 4 * sizeof(float)));
      points_capacity_ = count;
    }
    checkCudaErrors(
      cudaMemcpyAsync(
        points_dev_, packed_host_.data(), count * 4 * sizeof(float),
        cudaMemcpyHostToDevice, stream_));
    checkCudaErrors(cudaStreamSynchronize(stream_));   // stream, not the whole device

    pointpillar_->doinfer(points_dev_, static_cast<unsigned int>(count), detections_);
  }

  // Detector boxes -> tracker detections (class remap, centre z -> bottom z).
  // Filtering here rather than at publish time makes score_threshold mean the
  // same thing as the compiled SCORE_THRESH: a box below it never reaches the
  // tracker, so it cannot start a track or steal an association from a real
  // object.
  const double min_score = score_threshold_.load(std::memory_order_relaxed);
  std::vector<lidar_tracking::Detection> dets;
  dets.reserve(detections_.size());
  for (const auto & box : detections_) {
    if (box.score < min_score) {
      continue;
    }
    lidar_tracking::Detection d;
    if (to_detection(box, class_remap_, convert_center_z_to_bottom_, &d)) {
      dets.push_back(d);
    } else {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "detection with class id %d has no entry in class_remap; dropped", box.id);
    }
  }

  // Advance the tracker EVERY frame, including empty ones, so tracks age out.
  const std::vector<lidar_tracking::TrackedObject> tracks = tracker_.update(dets);

  std_msgs::msg::Header header;
  if (use_msg_stamp_) {
    header.stamp = msg->header.stamp;
  } else {
    header.stamp = now();
  }
  if (output_frame_id_.empty()) {
    header.frame_id = msg->header.frame_id;
    if (!header.frame_id.empty() && header.frame_id.front() == '/') {
      header.frame_id.erase(0, 1);   // ROS 2 frame ids carry no leading slash
    }
  } else {
    header.frame_id = output_frame_id_;
  }

  object_detection_msgs::msg::Object3dArray out;
  out.header = header;
  out.objects.reserve(tracks.size());
  for (const auto & t : tracks) {
    Eigen::Matrix<double, 7, 1> box;
    box << t.x, t.y, t.z, t.w, t.l, t.h, t.yaw;
    const lidar_tracking::Corners corners = lidar_tracking::bbox3d2corners(box);

    object_detection_msgs::msg::Object3d obj;
    obj.label = static_cast<uint8_t>(t.label);
    obj.confidence_score = static_cast<float>(t.score);
    obj.track_id = t.track_id;
    for (int i = 0; i < 8; ++i) {
      geometry_msgs::msg::Point c;
      c.x = corners(i, 0);
      c.y = corners(i, 1);
      c.z = corners(i, 2);
      obj.bounding_box.corners[i] = c;
    }
    out.objects.push_back(obj);
  }
  pub_objects_->publish(out);
  if (getenv("PP_TIME_CB")) {
    const auto t2 = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
      };
    RCLCPP_INFO(
      get_logger(), "CB: repack(%zu pts)=%.1f ms  infer+track+publish=%.1f ms  total=%.1f ms",
      count, ms(t0, t1), ms(t1, t2), ms(t0, t2));
  }

  if (publish_markers_ && pub_markers_) {
    publish_markers(tracks, header);
  }
}

void PointPillarsNode::validate_engine_against_params()
{
  // Every mismatch found during bring-up failed SILENTLY. An ONNX exported for
  // 10000 pillars, run against a params.h configured for 40000, produced ~70
  // confident-looking detections per frame that were pure noise -- nothing
  // crashed, nothing warned. These checks turn that class of failure into a
  // refusal to start.
  Params params;
  // Channel count comes from params.h (generated from the model config), not a
  // literal: OpenPCDet's PillarVFE builds 10, the mmdet3d-style model builds 9.
  constexpr int kPillarFeatures = PILLAR_FEATURES;

  const int head_h = params.grid_y_size / 2;   // feature_map_stride = 2
  const int head_w = params.grid_x_size / 2;
  const int num_anchors = Params::num_anchors;
  const int num_classes = Params::num_classes;

  std::vector<std::string> problems;

  auto want = [&](const char * name, std::initializer_list<int> expect) {
      const nvinfer1::Dims d = pointpillar_->tensorShape(name);
      if (d.nbDims < 0) {
        problems.push_back(std::string("engine has no tensor '") + name + "'");
        return;
      }
      const std::vector<int> exp(expect);
      if (static_cast<size_t>(d.nbDims) != exp.size()) {
        problems.push_back(
          std::string(name) + ": engine has " + std::to_string(d.nbDims) +
          " dims, params.h implies " + std::to_string(exp.size()));
        return;
      }
      for (size_t i = 0; i < exp.size(); ++i) {
        if (exp[i] >= 0 && d.d[i] != exp[i]) {
          std::string got, wanted;
          for (int k = 0; k < d.nbDims; ++k) {
            got += (k ? "," : "") + std::to_string(d.d[k]);
            wanted += (k ? "," : "") + (exp[k] < 0 ? std::string("*") : std::to_string(exp[k]));
          }
          problems.push_back(
            std::string(name) + ": engine [" + got + "] but params.h implies [" + wanted + "]");
          return;
        }
      }
    };

  want("voxels", {MAX_VOXELS, params.max_num_points_per_pillar, kPillarFeatures});
  want("voxel_idxs", {MAX_VOXELS, 4});
  want("cls_preds", {1, head_h, head_w, num_anchors * num_classes});
  want("box_preds", {1, head_h, head_w, num_anchors * 7});
  want("dir_cls_preds", {1, head_h, head_w, num_anchors * 2});

  if (static_cast<int>(class_remap_.size()) != num_classes) {
    problems.push_back(
      "class_remap has " + std::to_string(class_remap_.size()) + " entries but the "
      "model has " + std::to_string(num_classes) + " classes");
  }

  if (!problems.empty()) {
    RCLCPP_FATAL(
      get_logger(),
      "The loaded engine does not match params.h. This is almost always an ONNX "
      "exported with different settings than the runtime was built for -- it would "
      "otherwise run and emit plausible-looking nonsense. Regenerate params.h from "
      "the training config, or re-export the ONNX. Mismatches:");
    for (const auto & p : problems) {
      RCLCPP_FATAL(get_logger(), "  - %s", p.c_str());
    }
    throw std::runtime_error("engine/params.h mismatch; refusing to start");
  }
  RCLCPP_INFO(
    get_logger(),
    "engine matches params.h (%d classes, %d anchors, grid %dx%d, head %dx%d, "
    "max %d pillars)",
    num_classes, num_anchors, params.grid_x_size, params.grid_y_size,
    head_w, head_h, MAX_VOXELS);
}

void PointPillarsNode::publish_markers(
  const std::vector<lidar_tracking::TrackedObject> & tracks,
  const std_msgs::msg::Header & header)
{
  visualization_msgs::msg::MarkerArray arr;
  int id = 0;
  for (const auto & t : tracks) {
    visualization_msgs::msg::Marker m;
    m.header = header;
    m.ns = "objects";
    m.id = id++;
    m.type = visualization_msgs::msg::Marker::CUBE;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.position.x = t.x;
    m.pose.position.y = t.y;
    m.pose.position.z = t.z + t.h / 2.0;   // marker pose is the CENTRE
    // Rotation about z. The original hardcoded identity, and the commented-out
    // attempt put cos(rt/2) on .x, which is not a z-rotation at all.
    m.pose.orientation.x = 0.0;
    m.pose.orientation.y = 0.0;
    m.pose.orientation.z = std::sin(t.yaw / 2.0);
    m.pose.orientation.w = std::cos(t.yaw / 2.0);
    m.scale.x = t.w;
    m.scale.y = t.l;
    m.scale.z = t.h;
    m.color.a = 0.5;
    m.color.r = 1.0;
    m.color.g = 0.0;
    m.color.b = 0.0;
    arr.markers.push_back(m);
  }
  pub_markers_->publish(arr);
}

}  // namespace cuda_pointpillars_ros
