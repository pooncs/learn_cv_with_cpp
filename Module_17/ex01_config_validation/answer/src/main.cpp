#include "config.hpp"
#include <fmt/core.h>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        fmt::print(stderr, "Usage: {} <config_path>\n", argv[0]);
        return 1;
    }

    try {
        AppConfig config = ConfigLoader::load(argv[1]);
        fmt::print("Config loaded successfully!\n");
        fmt::print("Model: {}\n", config.model.name);
        fmt::print("Input Size: {}x{}\n", config.model.input_size[0], config.model.input_size[1]);
        fmt::print("Confidence: {}\n", config.model.confidence_threshold);
        fmt::print("Dataset: {}\n", config.dataset_path);
    } catch (const std::exception& e) {
        fmt::print(stderr, "Error loading config: {}\n", e.what());
        return 1;
    }

    return 0;
}
