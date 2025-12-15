#include <gtest/gtest.h>
#include <tl/expected.hpp>
#include <string>

tl::expected<int, std::string> divide(int a, int b) {
    if (b == 0) return tl::make_unexpected("Division by zero");
    return a / b;
}

TEST(ErrorTest, Expected) {
    auto res = divide(10, 2);
    EXPECT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 5);
    
    auto fail = divide(10, 0);
    EXPECT_FALSE(fail.has_value());
    EXPECT_EQ(fail.error(), "Division by zero");
}
