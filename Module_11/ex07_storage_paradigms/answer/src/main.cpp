#include <fmt/core.h>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

void benchmark_filesystem(int count, const fs::path& dir) {
    if (!fs::exists(dir)) fs::create_directory(dir);

    // Write
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < count; ++i) {
        std::ofstream out(dir / ("file_" + std::to_string(i) + ".bin"), std::ios::binary);
        std::string data(1024, 'A'); // 1KB
        out.write(data.c_str(), data.size());
    }
    auto end = std::chrono::high_resolution_clock::now();
    fmt::print("FS Write: {} ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    // Read
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < count; ++i) {
        std::ifstream in(dir / ("file_" + std::to_string(i) + ".bin"), std::ios::binary);
        std::string buffer(1024, '\0');
        in.read(&buffer[0], 1024);
    }
    end = std::chrono::high_resolution_clock::now();
    fmt::print("FS Read:  {} ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    
    // Cleanup
    fs::remove_all(dir);
}

void benchmark_aggregated(int count, const fs::path& file) {
    // Write
    auto start = std::chrono::high_resolution_clock::now();
    {
        std::ofstream out(file, std::ios::binary);
        std::string data(1024, 'A');
        for (int i = 0; i < count; ++i) {
            out.write(data.c_str(), data.size());
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    fmt::print("Agg Write: {} ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    // Read
    start = std::chrono::high_resolution_clock::now();
    {
        std::ifstream in(file, std::ios::binary);
        std::string buffer(1024, '\0');
        for (int i = 0; i < count; ++i) {
            in.read(&buffer[0], 1024);
        }
    }
    end = std::chrono::high_resolution_clock::now();
    fmt::print("Agg Read:  {} ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    
    fs::remove(file);
}

int main() {
    int count = 5000;
    fmt::print("Benchmarking {} items (1KB each)...\n", count);
    
    benchmark_filesystem(count, "data_lake_temp");
    benchmark_aggregated(count, "warehouse.bin");

    return 0;
}
