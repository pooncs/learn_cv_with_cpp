#include <gtest/gtest.h>
#include "normals.hpp"

TEST(NormalTest, PlaneZ) {
    std::vector<Point3D> cloud;
    // Plane Z=0
    for(int x=0; x<5; ++x) {
        for(int y=0; y<5; ++y) {
            cloud.push_back({(float)x, (float)y, 0.0f});
        }
    }
    
    auto normals = compute_normals(cloud, 5);
    // Center point (2,2,0) is at index 12
    auto n = normals[12];
    
    // Normal should be (0,0,1) or (0,0,-1)
    EXPECT_NEAR(std::abs(n.nz), 1.0f, 1e-4);
    EXPECT_NEAR(n.nx, 0.0f, 1e-4);
    EXPECT_NEAR(n.ny, 0.0f, 1e-4);
}
