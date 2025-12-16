#include <gtest/gtest.h>
#include "mesh.hpp"

TEST(MeshTest, GridTriangulation) {
    // 2x2 grid = 4 vertices. 1 quad. 2 triangles.
    auto faces = triangulate_grid(2, 2);
    EXPECT_EQ(faces.size(), 2);
    
    // 3x3 grid = 9 vertices. 4 quads. 8 triangles.
    faces = triangulate_grid(3, 3);
    EXPECT_EQ(faces.size(), 8);
}
