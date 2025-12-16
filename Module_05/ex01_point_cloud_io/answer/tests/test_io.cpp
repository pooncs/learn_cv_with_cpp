#include <gtest/gtest.h>
#include "PointCloudIO.hpp"
#include <cstdio>

TEST(IOTest, XYZ) {
    std::vector<Point3D> points = {{1, 2, 3}, {4, 5, 6}};
    write_xyz("test.xyz", points);
    auto loaded = read_xyz("test.xyz");
    ASSERT_EQ(loaded.size(), 2);
    EXPECT_FLOAT_EQ(loaded[0].x, 1);
    std::remove("test.xyz");
}

TEST(IOTest, PLY) {
    std::vector<Point3D> points = {{1, 2, 3}, {4, 5, 6}};
    write_ply("test.ply", points);
    auto loaded = read_ply("test.ply");
    ASSERT_EQ(loaded.size(), 2);
    EXPECT_FLOAT_EQ(loaded[1].y, 5);
    std::remove("test.ply");
}
