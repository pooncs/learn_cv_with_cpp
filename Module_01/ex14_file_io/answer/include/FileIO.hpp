#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>

struct Config {
    int width;
    int height;
    std::string app_name;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Config, width, height, app_name)
};

void save_binary(const std::string& filename, const std::vector<float>& data);
std::vector<float> load_binary(const std::string& filename);

void save_config(const std::string& filename, const Config& cfg);
Config load_config(const std::string& filename);
