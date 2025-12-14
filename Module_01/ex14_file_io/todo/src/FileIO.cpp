#include "FileIO.hpp"
#include <fstream>

void save_binary(const std::string& filename, const std::vector<float>& data) {
    // TODO: Write vector to binary file
}

std::vector<float> load_binary(const std::string& filename) {
    // TODO: Read vector from binary file
    return {};
}

void save_config(const std::string& filename, const Config& cfg) {
    // TODO: Serialize to JSON and save
}

Config load_config(const std::string& filename) {
    // TODO: Load JSON and deserialize
    return {};
}
