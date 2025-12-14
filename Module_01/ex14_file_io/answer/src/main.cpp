#include <iostream>
#include "FileIO.hpp"

int main() {
    // Binary
    std::vector<float> data = {1.1f, 2.2f, 3.3f};
    save_binary("test.bin", data);
    auto loaded_data = load_binary("test.bin");
    std::cout << "Loaded " << loaded_data.size() << " floats. First: " << loaded_data[0] << "\n";

    // JSON
    Config cfg{1920, 1080, "MyCVApp"};
    save_config("config.json", cfg);
    auto loaded_cfg = load_config("config.json");
    std::cout << "Loaded Config: " << loaded_cfg.app_name << " " << loaded_cfg.width << "x" << loaded_cfg.height << "\n";

    return 0;
}
