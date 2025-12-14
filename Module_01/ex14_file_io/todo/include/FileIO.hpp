#pragma once
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

struct Config {
    int width;
    int height;
    std::string app_name;

    // TODO: Add NLOHMANN_DEFINE_TYPE_INTRUSIVE macro
};

void save_binary(const std::string& filename, const std::vector<float>& data);
std::vector<float> load_binary(const std::string& filename);

void save_config(const std::string& filename, const Config& cfg);
Config load_config(const std::string& filename);
