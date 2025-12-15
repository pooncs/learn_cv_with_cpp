#include "cache.hpp"
#include <fmt/core.h>
#include <thread>
#include <chrono>

std::string expensive_op(const std::string& input) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return "Processed: " + input;
}

int main() {
    CacheManager cache("my_cache");
    std::string input = "image_01.png";
    std::string hash = cache.compute_hash(input);

    if (cache.exists(hash)) {
        fmt::print("Hit! {}\n", cache.load(hash));
    } else {
        fmt::print("Miss! Computing...\n");
        std::string result = expensive_op(input);
        cache.save(hash, result);
        fmt::print("Done. {}\n", result);
    }
    return 0;
}
