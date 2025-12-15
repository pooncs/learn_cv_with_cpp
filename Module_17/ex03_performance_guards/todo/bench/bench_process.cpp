#include <benchmark/benchmark.h>
#include "../src/image_proc.hpp"

static void BM_ProcessImage(benchmark::State& state) {
    // TODO: Write a benchmark loop
    // 1. Loop over state
    // 2. Call process_image(640, 480)
    // 3. Update items processed using state.SetItemsProcessed()
    for (auto _ : state) {
        // ...
    }
}
BENCHMARK(BM_ProcessImage);

BENCHMARK_MAIN();
