#include <gtest/gtest.h>
#include "octree.hpp"

TEST(OctreeTest, RadiusSearch) {
    AABB bounds{{0, 0, 0}, {10, 10, 10}};
    Octree tree(bounds, 2); // Small capacity to force subdivision
    
    tree.insert({1, 1, 1});
    tree.insert({2, 2, 2});
    tree.insert({8, 8, 8}); // Far away
    
    auto res = tree.query_radius({1.5, 1.5, 1.5}, 2.0);
    
    // Should find (1,1,1) and (2,2,2). Distance to (1,1,1) is sqrt(0.75) < 2.
    // Distance to (8,8,8) is large.
    EXPECT_EQ(res.size(), 2);
}
