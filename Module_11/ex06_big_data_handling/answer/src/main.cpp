#include <fmt/core.h>
#include <vector>
#include <string>
#include <numeric>
#include <thread>
#include <future>
#include <algorithm>

// Simulate processing a chunk of data
long process_chunk(const std::vector<std::string>& chunk) {
    long total_chars = 0;
    for (const auto& str : chunk) {
        total_chars += str.length();
        // Simulate heavy work
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return total_chars;
}

int main() {
    // Generate dummy data: 1000 strings
    std::vector<std::string> dataset;
    for (int i = 0; i < 1000; ++i) {
        dataset.push_back("Data item " + std::to_string(i));
    }

    // Split into shards (Map phase)
    int num_shards = 4;
    std::vector<std::future<long>> futures;
    size_t chunk_size = dataset.size() / num_shards;

    fmt::print("Processing {} items with {} threads...\n", dataset.size(), num_shards);

    for (int i = 0; i < num_shards; ++i) {
        auto start_it = dataset.begin() + i * chunk_size;
        auto end_it = (i == num_shards - 1) ? dataset.end() : start_it + chunk_size;
        std::vector<std::string> chunk(start_it, end_it);

        futures.push_back(std::async(std::launch::async, process_chunk, chunk));
    }

    // Reduce phase
    long total_result = 0;
    for (auto& f : futures) {
        total_result += f.get();
    }

    fmt::print("Total characters: {}\n", total_result);
    return 0;
}
