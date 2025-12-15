#include <gtest/gtest.h>
#include <vector>
#include <list>
#include <numeric>

// Simple test to ensure STL containers are used
TEST(STLPerfTest, VectorSum) {
    std::vector<int> v(1000, 1);
    int sum = std::accumulate(v.begin(), v.end(), 0);
    EXPECT_EQ(sum, 1000);
}
