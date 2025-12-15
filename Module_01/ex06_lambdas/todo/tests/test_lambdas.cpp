#include <gtest/gtest.h>
#include <algorithm>
#include <vector>

TEST(LambdaTest, Sort) {
    std::vector<int> v = {3, 1, 4, 1, 5};
    std::sort(v.begin(), v.end(), [](int a, int b){
        return a > b; // Descending
    });
    EXPECT_EQ(v[0], 5);
    EXPECT_EQ(v[4], 1);
}
