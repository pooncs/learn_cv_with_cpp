#include "factory.hpp"
#include <fmt/core.h>

int main() {
    auto& factory = AlgorithmFactory::instance();

    // Register algorithms
    factory.register_algo("Canny", []() { return std::make_unique<CannyDetector>(); });
    factory.register_algo("Sobel", []() { return std::make_unique<SobelDetector>(); });

    try {
        // Create and use Canny
        auto algo1 = factory.create("Canny");
        algo1->process({"Image_01"});

        // Create and use Sobel
        auto algo2 = factory.create("Sobel");
        algo2->process({"Image_02"});

        // Try unknown
        auto algo3 = factory.create("YOLO");
    } catch (const std::exception& e) {
        fmt::print(stderr, "Error: {}\n", e.what());
    }

    return 0;
}
