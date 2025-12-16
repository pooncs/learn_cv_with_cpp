#include <benchmark/benchmark.h>
#include <vector>
#include <cstdlib>
#include <algorithm>

// Simulate simple thresholding
void threshold(std::vector<uint8_t>& data, uint8_t thresh) {
    for(auto& px : data) {
        if(px > thresh) px = 255;
        else px = 0;
    }
}

static void BM_ProcessRandom(benchmark::State& state) {
    int size = state.range(0);
    std::vector<uint8_t> data(size);
    // Fill with random
    for(auto& px : data) px = rand() % 256;

    for (auto _ : state) {
        // Copy to avoid modifying the source for next iter (though strictly we re-threshold)
        // For benchmark we just process in place, it will settle to 0/255 quickly, 
        // effectively benchmarking branch prediction on binary data.
        // To be rigorous we should copy.
        std::vector<uint8_t> work = data;
        threshold(work, 128);
        benchmark::DoNotOptimize(work.data());
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
}
BENCHMARK(BM_ProcessRandom)->Range(1<<10, 1<<20);

static void BM_ProcessGradient(benchmark::State& state) {
    int size = state.range(0);
    std::vector<uint8_t> data(size);
    // Fill with gradient (sorted data effectively)
    for(int i=0; i<size; ++i) data[i] = (i * 255) / size;

    for (auto _ : state) {
        std::vector<uint8_t> work = data;
        threshold(work, 128);
        benchmark::DoNotOptimize(work.data());
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(state.range(0)));
}
BENCHMARK(BM_ProcessGradient)->Range(1<<10, 1<<20);

BENCHMARK_MAIN();
