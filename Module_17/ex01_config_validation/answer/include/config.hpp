#pragma once
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

struct ModelConfig {
    std::string name;
    std::vector<int> input_size;
    float confidence_threshold;
};

struct AppConfig {
    ModelConfig model;
    std::string dataset_path;
};

class ConfigLoader {
public:
    static AppConfig load(const std::string& path);
private:
    static void validate(const YAML::Node& node);
};
