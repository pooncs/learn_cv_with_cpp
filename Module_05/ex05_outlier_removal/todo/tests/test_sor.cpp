#include <gtest/gtest.h>
#include "sor.hpp"

TEST(SORTest, Outlier) {
    std::vector<Point3D> cloud;
    for(int i=0; i<10; ++i) cloud.push_back({0, 0, 0}); // Cluster
    cloud.push_back({100, 100, 100}); // Outlier
    
    // k=5. Cluster points dist ~ 0. Outlier dist ~ large.
    auto res = remove_outliers(cloud, 5, 1.0f);
    
    // Outlier should be removed
    EXPECT_EQ(res.size(), 10);
    for(auto p : res) {
        EXPECT_NEAR(p.x, 0.0f, 1e-4);
    }
}
