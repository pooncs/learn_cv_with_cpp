#include "simd_ops.hpp"
#include <benchmark/benchmark.h>
#include <vector>
#include <random>

static std::vector<float> A, B;

void SetupData() {
    if (A.empty()) {
        size_t n = 1000000;
        A.resize(n);
        B.resize(n);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for(size_t i=0; i<n; ++i) {
            A[i] = dist(rng);
            B[i] = dist(rng);
        }
    }
}

static void BM_Scalar(benchmark::State& state) {
    SetupData();
    for (auto _ : state) {
        benchmark::DoNotOptimize(dot_product_scalar(A.data(), B.data(), A.size()));
    }
}
BENCHMARK(BM_Scalar);

static void BM_AVX2(benchmark::State& state) {
    SetupData();
    for (auto _ : state) {
        benchmark::DoNotOptimize(dot_product_avx2(A.data(), B.data(), A.size()));
    }
}
BENCHMARK(BM_AVX2);

BENCHMARK_MAIN();
