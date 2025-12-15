#include <gtest/gtest.h>
#include <tuple>

struct Point {
    int x, y;
};

TEST(BindingTest, StructBinding) {
    Point p{10, 20};
    auto [x, y] = p;
    EXPECT_EQ(x, 10);
    EXPECT_EQ(y, 20);
}
