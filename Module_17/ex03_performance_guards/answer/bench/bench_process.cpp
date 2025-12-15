#include <benchmark/benchmark.h>
#include "../src/image_proc.hpp"

static void BM_ProcessImage(benchmark::State& state) {
    for (auto _ : state) {
        process_image(640, 480);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ProcessImage)->UseRealTime();

BENCHMARK_MAIN();
