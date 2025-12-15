#pragma once
#include <immintrin.h>
#include <cstddef>

inline float dot_product_scalar(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline float dot_product_avx2(const float* a, const float* b, size_t n) {
    // TODO: Implement AVX2 dot product
    // 1. Initialize zero vector: _mm256_setzero_ps()
    // 2. Loop with stride 8
    // 3. Load: _mm256_loadu_ps
    // 4. FMA: _mm256_fmadd_ps
    // 5. Horizontal sum
    // 6. Handle tail
    
    return dot_product_scalar(a, b, n); // Fallback
}
