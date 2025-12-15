#include "config.hpp"
#include <stdexcept>
#include <fmt/core.h>

void ConfigLoader::validate(const YAML::Node& node) {
    // TODO: Check if 'model' section exists
    
    // TODO: Check if 'dataset' section exists
    
    // TODO: Validate model parameters (name, input_size, confidence_threshold)
    // Hint: Check ranges and types.
}

AppConfig ConfigLoader::load(const std::string& path) {
    YAML::Node node = YAML::LoadFile(path);
    
    // TODO: Call validate(node)
    
    AppConfig config;
    // TODO: Parse the YAML node into the config struct
    
    return config;
}
