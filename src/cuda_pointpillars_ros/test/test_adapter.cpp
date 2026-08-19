// Pins the detector -> tracker boundary. These conversions do not crash when
// wrong; they produce plausible-looking, quietly incorrect output. That is what
// makes them worth a test.
#include "cuda_pointpillars_ros/cuda_pp_ros.hpp"

#include <gtest/gtest.h>

using cuda_pointpillars_ros::find_point_fields;
using cuda_pointpillars_ros::to_detection;

namespace
{
// Detector emits KITTI order {Car 0, Pedestrian 1, Cyclist 2}; the pipeline uses
// {Pedestrian 0, Cyclist 1, Car 2}.
const std::vector<int64_t> kKittiRemap{2, 0, 1};
// A 2-class Livox model emits {Pedestrian 0, Car 1}; CYCLIST=1 stays reserved.
const std::vector<int64_t> kLivoxRemap{0, 2};

Bndbox car()
{
  // KITTI Car anchor: w is the ALONG-HEADING extent (3.9), z is a centre.
  return Bndbox(12.0f, -3.0f, -1.0f, 3.9f, 1.6f, 1.56f, 0.25f, /*id=*/0, 0.9f);
}
}  // namespace

TEST(Adapter, KittiCarMapsToCarLabel)
{
  lidar_tracking::Detection d;
  ASSERT_TRUE(to_detection(car(), kKittiRemap, /*convert_z=*/false, &d));
  EXPECT_EQ(d.label, lidar_tracking::kCar) << "detector class 0 (Car) must become label 2";
}

TEST(Adapter, DimensionsPassThroughUnswapped)
{
  lidar_tracking::Detection d;
  ASSERT_TRUE(to_detection(car(), kKittiRemap, false, &d));
  // Bndbox.w is the along-heading length and bbox3d2corners' dims[0] is the x
  // extent, so these copy across with NO swap. Getting this wrong makes every
  // car 1.6 m long and 3.9 m wide.
  // Bndbox stores float and the adapter widens to double, so compare at float
  // precision rather than exactly.
  EXPECT_NEAR(d.w, 3.9, 1e-6);
  EXPECT_NEAR(d.l, 1.6, 1e-6);
  EXPECT_NEAR(d.h, 1.56, 1e-6);
  EXPECT_NEAR(d.yaw, 0.25, 1e-6);
  EXPECT_NEAR(d.score, 0.9, 1e-6);
}

TEST(Adapter, ZConventionIsOffByDefault)
{
  lidar_tracking::Detection d;
  // Default reproduces the validated pipeline: the decoded centre z is passed
  // through untouched, matching what the IVI has always consumed.
  ASSERT_TRUE(to_detection(car(), kKittiRemap, false, &d));
  EXPECT_NEAR(d.z, -1.0, 1e-6);

  // Enabled, it converts centre -> bottom, which is geometrically correct.
  ASSERT_TRUE(to_detection(car(), kKittiRemap, true, &d));
  EXPECT_NEAR(d.z, -1.0 - 1.56 / 2.0, 1e-6);
}

TEST(Adapter, FullKittiRemapIsAPermutation)
{
  const int expected[3] = {lidar_tracking::kCar, lidar_tracking::kPedestrian,
    lidar_tracking::kCyclist};
  for (int detector_id = 0; detector_id < 3; ++detector_id) {
    Bndbox b = car();
    b.id = detector_id;
    lidar_tracking::Detection d;
    ASSERT_TRUE(to_detection(b, kKittiRemap, false, &d));
    EXPECT_EQ(d.label, expected[detector_id]) << "detector class " << detector_id;
  }
}

TEST(Adapter, TwoClassLivoxModelSkipsCyclist)
{
  Bndbox ped = car(); ped.id = 0;
  Bndbox veh = car(); veh.id = 1;
  lidar_tracking::Detection d;

  ASSERT_TRUE(to_detection(ped, kLivoxRemap, false, &d));
  EXPECT_EQ(d.label, lidar_tracking::kPedestrian);
  ASSERT_TRUE(to_detection(veh, kLivoxRemap, false, &d));
  EXPECT_EQ(d.label, lidar_tracking::kCar) << "CYCLIST=1 stays reserved and unused";
}

TEST(Adapter, OutOfRangeClassIsRejectedNotWrapped)
{
  Bndbox b = car();
  b.id = 7;
  lidar_tracking::Detection d;
  EXPECT_FALSE(to_detection(b, kKittiRemap, false, &d));
  b.id = -1;
  EXPECT_FALSE(to_detection(b, kKittiRemap, false, &d));
}

// ── point field lookup ──────────────────────────────────────────────────────
namespace
{
sensor_msgs::msg::PointCloud2 cloud_with(
  const std::vector<std::pair<std::string, uint32_t>> & fields, uint32_t point_step)
{
  sensor_msgs::msg::PointCloud2 m;
  m.point_step = point_step;
  for (const auto & [name, offset] : fields) {
    sensor_msgs::msg::PointField f;
    f.name = name;
    f.offset = offset;
    f.datatype = sensor_msgs::msg::PointField::FLOAT32;
    f.count = 1;
    m.fields.push_back(f);
  }
  return m;
}
}  // namespace

TEST(PointFields, FindsKittiLayoutWithGapAndStride32)
{
  // The real KITTI bag: point_step 32, intensity at offset 16 -- NOT the packed
  // 16-byte float4 the preprocessor ultimately wants. Offsets must be read by
  // name, never assumed.
  const auto m = cloud_with({{"x", 0}, {"y", 4}, {"z", 8}, {"intensity", 16}}, 32);
  const auto o = find_point_fields(m);
  ASSERT_TRUE(o.valid());
  EXPECT_EQ(o.x, 0);
  EXPECT_EQ(o.intensity, 16);
}

TEST(PointFields, AcceptsReflectivityAsIntensity)
{
  // Livox / RoboSense drivers name the 4th channel "reflectivity".
  const auto m = cloud_with({{"x", 0}, {"y", 4}, {"z", 8}, {"reflectivity", 12}, {"tag", 16}}, 18);
  const auto o = find_point_fields(m);
  ASSERT_TRUE(o.valid());
  EXPECT_EQ(o.intensity, 12);
}

TEST(PointFields, RejectsCloudMissingIntensity)
{
  const auto m = cloud_with({{"x", 0}, {"y", 4}, {"z", 8}}, 12);
  EXPECT_FALSE(find_point_fields(m).valid());
}
