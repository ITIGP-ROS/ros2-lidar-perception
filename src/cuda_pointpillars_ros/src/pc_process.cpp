#include "cuda_pointpillars_ros/cuda_pp_ros.hpp"

#include <rclcpp/rclcpp.hpp>

#include <cstdio>
#include <exception>
#include <memory>

int main(int argc, char ** argv)
{
  cuda_pointpillars_ros::Getinfo();
  rclcpp::init(argc, argv);

  try {
    rclcpp::spin(std::make_shared<cuda_pointpillars_ros::PointPillarsNode>());
  } catch (const std::exception & e) {
    // Startup validation (engine vs params.h) and a missing model_path both
    // throw. Exit with a status rather than letting the exception escape into
    // std::terminate and dump core -- the message is the useful part.
    RCLCPP_FATAL(rclcpp::get_logger("cuda_pointpillars_node"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
