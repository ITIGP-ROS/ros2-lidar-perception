cmake_minimum_required(VERSION 3.8)
project(object_detection_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BoundingBox3d.msg"
  "msg/Object3d.msg"
  "msg/Object3dArray.msg"
  DEPENDENCIES geometry_msgs # Add packages that above messages depend on, in this case geometry_msgs for Sphere.msg
)

if(BUILD_TESTING)
  # Optional rather than REQUIRED: ament_lint_auto is not installed everywhere
  # (it is absent on this workstation and on a minimal Yocto SDK), and an
  # unconditional REQUIRED made the whole package -- and therefore the whole
  # workspace -- unbuildable with testing enabled.
  find_package(ament_lint_auto QUIET)
  if(NOT ament_lint_auto_FOUND)
    message(STATUS "ament_lint_auto not found; skipping lint tests")
    return()
  endif()
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
