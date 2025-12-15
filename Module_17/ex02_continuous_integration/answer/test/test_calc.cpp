#include <gtest/gtest.h>
#include "../src/calc.hpp"

TEST(CalcTest, Add) {
    EXPECT_EQ(add(2, 3), 5);
    EXPECT_EQ(add(-1, 1), 0);
}
