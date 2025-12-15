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
    __m256 sum_vec = _mm256_setzero_ps();
    size_t i = 0;

    // Process 8 floats at a time
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
#ifdef __FMA__
        sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);
#else
        sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(va, vb));
#endif
    }

    // Horizontal sum
    float temp[8];
    _mm256_storeu_ps(temp, sum_vec);
    float sum = 0.0f;
    for (int j = 0; j < 8; ++j) sum += temp[j];

    // Handle tail
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}
