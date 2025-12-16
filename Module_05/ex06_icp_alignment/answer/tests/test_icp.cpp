#include <gtest/gtest.h>
#include "icp.hpp"

TEST(ICPTest, Translation) {
    std::vector<Point3D> target = {{0,0,0}, {1,0,0}, {0,1,0}};
    std::vector<Point3D> source = {{1,0,0}, {2,0,0}, {1,1,0}}; // Shifted by +1 in X
    
    // T should be X-1
    Eigen::Matrix4f T = icp_align(source, target, 5);
    
    // T * source ~ target
    // We want the transform that moves source to target. So T should have translation -1 in X.
    EXPECT_NEAR(T(0, 3), -1.0f, 1e-4);
    EXPECT_NEAR(T(1, 3), 0.0f, 1e-4);
    EXPECT_NEAR(T(2, 3), 0.0f, 1e-4);
}
