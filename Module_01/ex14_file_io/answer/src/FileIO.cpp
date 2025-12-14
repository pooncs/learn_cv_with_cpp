#include "FileIO.hpp"
#include <fstream>
#include <iostream>

void save_binary(const std::string& filename, const std::vector<float>& data) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open file for writing");
    // Write size first
    size_t size = data.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    // Write data
    out.write(reinterpret_cast<const char*>(data.data()), size * sizeof(float));
}

std::vector<float> load_binary(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file for reading");
    
    size_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    
    std::vector<float> data(size);
    in.read(reinterpret_cast<char*>(data.data()), size * sizeof(float));
    return data;
}

void save_config(const std::string& filename, const Config& cfg) {
    std::ofstream out(filename);
    nlohmann::json j = cfg;
    out << j.dump(4);
}

Config load_config(const std::string& filename) {
    std::ifstream in(filename);
    nlohmann::json j;
    in >> j;
    return j.get<Config>();
}
