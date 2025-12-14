#include <gtest/gtest.h>
#include "MathUtils.hpp"

TEST(MathUtilsTest, Factorial) {
    EXPECT_EQ(MathUtils::factorial(0), 1);
    EXPECT_EQ(MathUtils::factorial(1), 1);
    EXPECT_EQ(MathUtils::factorial(5), 120);
    EXPECT_THROW(MathUtils::factorial(-1), std::invalid_argument);
}

TEST(MathUtilsTest, Fibonacci) {
    EXPECT_EQ(MathUtils::fibonacci(0), 0);
    EXPECT_EQ(MathUtils::fibonacci(1), 1);
    EXPECT_EQ(MathUtils::fibonacci(6), 8); // 0 1 1 2 3 5 8
}

TEST(MathUtilsTest, Lerp) {
    EXPECT_DOUBLE_EQ(MathUtils::lerp(0.0, 10.0, 0.5), 5.0);
    EXPECT_DOUBLE_EQ(MathUtils::lerp(0.0, 10.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(MathUtils::lerp(0.0, 10.0, 1.0), 10.0);
}
