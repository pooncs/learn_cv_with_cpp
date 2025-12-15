#include <gtest/gtest.h>
#include "simd_ops.hpp"
#include <vector>
#include <cmath>

TEST(SIMDTest, Correctness) {
    size_t n = 1024 + 3; // +3 to test tail handling
    std::vector<float> a(n, 1.5f);
    std::vector<float> b(n, 2.0f);

    float expected = 0;
    for(size_t i=0; i<n; ++i) expected += a[i] * b[i];

    float result = dot_product_avx2(a.data(), b.data(), n);
    
    EXPECT_NEAR(result, expected, 1e-4);
}
