#include <gtest/gtest.h>
#include "async_tasks.hpp"

TEST(AsyncTest, RunsInParallel) {
    // 50 + 20 + 80 = 150ms sequential
    // Parallel max = 80ms
    auto res = cv_curriculum::processFrameAsync(10);
    
    // If it takes < 140ms, it's likely parallel
    // If it takes > 140ms, it's sequential (or machine is very slow)
    EXPECT_LT(res.totalTimeMs, 145.0);
    EXPECT_GT(res.totalTimeMs, 75.0); // Can't be faster than longest task
}
