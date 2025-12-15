#include <gtest/gtest.h>
#include "optimizer.hpp"

TEST(LeastSquaresTest, PerfectData) {
    // y = 1*x^2 + 2*x + 1
    std::vector<DataPoint> data;
    for (double x = -1.0; x <= 1.0; x += 0.5) {
        data.push_back({x, 1.0*x*x + 2.0*x + 1.0});
    }

    auto result = CurveFitter::solve(data);
    EXPECT_NEAR(result[0], 1.0, 1e-4);
    EXPECT_NEAR(result[1], 2.0, 1e-4);
    EXPECT_NEAR(result[2], 1.0, 1e-4);
}
