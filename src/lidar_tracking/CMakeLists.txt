cmake_minimum_required(VERSION 3.8)
project(lidar_tracking)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/types.cpp
  src/geometry.cpp
  src/metrics.cpp
  src/kalman.cpp
  src/assignment.cpp
  src/ab3dmot.cpp
  src/track3d.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(nlohmann_json QUIET)

  if(nlohmann_json_FOUND OR EXISTS /usr/include/nlohmann/json.hpp)
    ament_add_gtest(test_goldens test/test_goldens.cpp test/test_assignment.cpp)
    target_link_libraries(test_goldens ${PROJECT_NAME})
    target_compile_definitions(test_goldens PRIVATE
      GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/data")
    if(nlohmann_json_FOUND)
      target_link_libraries(test_goldens nlohmann_json::nlohmann_json)
    endif()
  else()
    message(WARNING "nlohmann/json not found -- golden tests are NOT built")
  endif()
endif()

ament_package()
