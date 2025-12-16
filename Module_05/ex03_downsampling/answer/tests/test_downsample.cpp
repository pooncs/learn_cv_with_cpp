#include <gtest/gtest.h>
#include "downsample.hpp"

TEST(DownsampleTest, Voxel) {
    std::vector<Point3D> cloud;
    // 10 points at (0,0,0) ... (0.9, 0, 0)
    for(int i=0; i<10; ++i) cloud.push_back({i*0.1f, 0, 0});
    
    // Voxel size 1.0 -> All should fall in same voxel (0,0,0)
    // Avg x = 0.45
    auto res = voxel_grid_downsample(cloud, 1.0f);
    ASSERT_EQ(res.size(), 1);
    EXPECT_NEAR(res[0].x, 0.45f, 1e-4);
}
