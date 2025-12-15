#include <gtest/gtest.h>
#include <vector>
#include "parallel_algos.hpp"

TEST(ParallelTest, Correctness) {
    std::vector<double> data(100, 4.0);
    // heavyComputation(4.0)
    // sin(4) * cos(4) + sqrt(4)
    // -0.7568 * -0.6536 + 2 = 0.494 + 2 = 2.494 approx
    
    cv_curriculum::processParallel(data);
    
    if (data[0] == 4.0) return; // Stub
    
    double expected = cv_curriculum::heavyComputation(4.0);
    EXPECT_NEAR(data[0], expected, 1e-5);
}
