#include <benchmark/benchmark.h>
#include <vector>

// TODO: Implement BM_VectorPush
// 1. Get size from state.range(0)
// 2. Loop state
// 3. Inside loop: create vector, push_back N times

// TODO: Implement BM_VectorReserve
// Same as above, but call v.reserve(N) first

// TODO: Register benchmarks with Range(8, 8<<10)

BENCHMARK_MAIN();
