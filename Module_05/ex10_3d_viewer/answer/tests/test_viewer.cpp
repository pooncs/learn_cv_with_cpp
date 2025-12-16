#include <gtest/gtest.h>
#include "viewer.hpp"

TEST(ViewerTest, Projection) {
    Camera cam(100, 100, 90.0f);
    // Look from (0,0,0) to (0,0,-1). Up (0,1,0).
    cam.lookAt({0,0,0}, {0,0,-1}, {0,1,0});
    
    // Point at (0,0,-10). Should project to center (50, 50).
    std::vector<Point3D> cloud = {{0, 0, -10, 255, 0, 0}};
    
    cv::Mat img = render_point_cloud(cloud, cam);
    
    // Check if pixel at (50,50) is red
    cv::Vec3b p = img.at<cv::Vec3b>(50, 50);
    // BGR
    EXPECT_EQ(p[2], 255);
    EXPECT_EQ(p[0], 0);
}
