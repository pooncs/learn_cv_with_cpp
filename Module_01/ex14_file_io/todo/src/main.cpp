#include <iostream>
#include "FileIO.hpp"

int main() {
    // Binary
    std::vector<float> data = {1.1f, 2.2f, 3.3f};
    // save_binary("test.bin", data);
    
    // JSON
    Config cfg{1920, 1080, "MyCVApp"};
    // save_config("config.json", cfg);

    return 0;
}
